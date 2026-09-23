"""Evaluate the three Hydra measurements for one robot.

ADD is the mean distance between ground-truth and predicted FK link origins in
camera coordinates. The summary reports ADD AUC at 100 mm.
"""
import logging
from pathlib import Path
from typing import Optional

# This import must precede torch and cv2 for deterministic environment settings.
from eval_utils import (
    seed_everything, build_eval_stack, compute_add_metrics, EstimatorAdapter,
    aggregate_frame_diagnostics, log_diagnostics_summary, _to_np,
    load_detections_for_run, run_frame_inference, save_eval_results,
    verify_detection_coverage, assert_translation_units_consistent,
    select_eval_frame_indices,
)

import numpy as np
import torch
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from loaders.hydra_loader import HydraDataset

log = logging.getLogger(__name__)


def _eval_measurement(
    model: EstimatorAdapter,
    dataset: HydraDataset,
    use_gpu: bool,
    measurement_idx: int,
    return_per_frame: bool,
    detections: dict,
    detection_ids: set = None,
    robot_tag: str = "",
    num_eval_frames: Optional[int] = None,
) -> tuple[list, list, list]:
    """
    Evaluate one Hydra measurement (15 frames, or num_eval_frames evenly-spaced
    frames when set - the limit applies per measurement).

    Returns (err_raw_m, queries_m, diags_m):
        err_raw_m  - per-frame ADD errors in metres
        queries_m  - per-frame dicts (or []) when return_per_frame=False
        diags_m    - per-frame diagnostics dicts
    """
    T_gt = dataset.T_cam_base          # (4,4) float64
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    # Stored per query so results JSONs are self-contained across datasets
    # (visualize_results.py & co. need K / GT without re-reading the data folder).
    gt_pose_list = T_gt.tolist()
    K_eval_list  = dataset.K.astype(float).tolist()

    err_raw: list   = []
    queries: list   = []
    diags: list     = []

    _meas_ids = [f"meas{measurement_idx}_{fi:03d}" for fi, _ in dataset._frames]
    # Detection-aware subsetting: an int num_eval_frames is drawn from the frames the
    # detections file covers (a no-op for full-coverage files) - see
    # eval_utils.select_eval_frame_indices.
    _covered = None
    if detection_ids:
        _covered = [i for i, fid in enumerate(_meas_ids) if fid in detection_ids] or None
    frame_indices = select_eval_frame_indices(len(dataset), num_eval_frames, _covered)
    _subset_src = f" of {len(_covered)} detection-covered frames" if _covered is not None else ""
    log.info(f"  meas_{measurement_idx}: evaluating {len(frame_indices)} of {len(dataset)} frames"
             + (f" (evenly-spaced subset{_subset_src})" if len(frame_indices) < len(dataset)
                else " (full dataset)"))

    verify_detection_coverage(
        detections or {}, detection_ids or set(),
        [_meas_ids[i] for i in frame_indices],
        f"meas{measurement_idx}",
    )

    for i in tqdm(frame_indices, desc=f"  meas_{measurement_idx}", leave=False):
        image, joint_angles, _, frame_idx = dataset[i]
        img_path = str(dataset._image_dir / f"{frame_idx:03d}.png")
        if use_gpu:
            image = image.cuda()
        joints_cpu = joint_angles.cpu().squeeze()

        cTr, diag = run_frame_inference(
            model, image, joints_cpu, f"meas{measurement_idx}_{frame_idx:03d}",
            detections, detection_ids, frame_key=img_path,
            K=dataset.K,  # this measurement's K, not measurement_0's
        )
        diags.append(diag)

        # FK link origins in robot base frame (metres).
        _, t_links = model.estimator.robot_kin.get_joint_R_t(joints_cpu)
        t_links = np.asarray(t_links, dtype=np.float64)     # (N_links, 3)

        # GT link positions in camera frame.
        kps_gt = (R_gt @ t_links.T + t_gt[:, None]).T       # (N_links, 3)

        if cTr is None:
            log.debug("Frame %03d meas_%d: PnP failed.", frame_idx, measurement_idx)
            err_raw.append(1.0)
            if return_per_frame:
                queries.append({
                    "measurement": measurement_idx,
                    "frame_idx":   frame_idx,
                    "frame_id":    f"meas{measurement_idx}_{frame_idx:03d}",
                    "image_path":  img_path,
                    "image_size_hw": [image.shape[-2], image.shape[-1]],
                    "K_eval":      K_eval_list,
                    "joints_rad":  joints_cpu.numpy().tolist(),
                    "gt_pose":     gt_pose_list,
                    "est_pose":    None,
                    "add_m_mm":    1000.0,
                    "pnp_failed":  True,
                    "diagnostics": diag,
                })
            continue

        cTr_np = _to_np(cTr).astype(np.float64).squeeze()
        R_pred = cTr_np[:3, :3]
        t_pred = cTr_np[:3, 3]

        # First-valid-frame guard: predicted base translation (t_pred) and GT base
        # translation (t_gt) are both the base origin in the camera frame, so they must
        # agree in unit. Catches a silent mesh-mm vs GT-metres mismatch (~1000×).
        assert_translation_units_consistent(t_pred, t_gt, tag=f"hydra_{robot_tag}")

        kps_pred = (R_pred @ t_links.T + t_pred[:, None]).T  # (N_links, 3)
        err      = float(np.mean(np.linalg.norm(kps_pred - kps_gt, axis=1)))
        err_raw.append(err)

        if return_per_frame:
            queries.append({
                "measurement": measurement_idx,
                "frame_idx":   frame_idx,
                "frame_id":    f"meas{measurement_idx}_{frame_idx:03d}",
                "image_path":  img_path,
                "image_size_hw": [image.shape[-2], image.shape[-1]],
                "K_eval":      K_eval_list,
                "joints_rad":  joints_cpu.numpy().tolist(),
                "gt_pose":     gt_pose_list,
                "est_pose":    cTr_np.tolist(),
                "add_m_mm":    round(err * 1000, 3),
                "pnp_failed":  False,
                "diagnostics": diag,
            })

    return err_raw, queries, diags


@hydra.main(config_path="../configs", config_name="eval_hydra_lbr", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.get("seed", 0))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = Path(cfg.dataset.data_folder)
    scale       = float(cfg.dataset.get("scale", 1.0))

    # Intrinsics: load native K from measurement_0, then scale to match images.
    K_native = np.load(data_folder / "measurement_0" / "camera_K.npy").astype(np.float32)
    K_eval_np = K_native.copy()
    if scale != 1.0:
        K_eval_np[0, 0] *= scale
        K_eval_np[0, 2] *= scale
        K_eval_np[1, 1] *= scale
        K_eval_np[1, 2] *= scale
    eval_h = int(round(cfg.robot.image_height * scale))
    eval_w = int(round(cfg.robot.image_width  * scale))

    adapter, model_stats = build_eval_stack(cfg, device, K_eval_np, (eval_h, eval_w))

    log.info("Robot: %s  |  data: %s  |  scale=%.2f -> %dx%d",
             cfg.robot.name, data_folder, scale, eval_w, eval_h)

    detections, detection_ids, det_meta = load_detections_for_run(cfg)

    return_per_frame = cfg.get("save_results", None) is not None
    use_gpu          = torch.cuda.is_available()

    # Evaluate all three measurements and pool errors.
    all_err_raw: list = []
    all_queries: list = []
    all_diags:   list = []

    for m in range(3):
        meas_dir = data_folder / f"measurement_{m}"
        if not meas_dir.exists():
            log.warning("measurement_%d not found - skipping", m)
            continue

        dataset = HydraDataset(meas_dir, scale=scale, trans_to_tensor=transforms.ToTensor())
        log.info("  measurement_%d: %d frames", m, len(dataset))

        errs, qs, diags = _eval_measurement(
            adapter, dataset, use_gpu,
            measurement_idx=m,
            return_per_frame=return_per_frame,
            detections=detections,
            detection_ids=detection_ids,
            robot_tag=cfg.robot.name,
            num_eval_frames=cfg.get("num_eval_frames", None),
        )
        all_err_raw.extend(errs)
        all_queries.extend(qs)
        all_diags.extend(diags)

        m_mm = np.array(errs) * 1000.0
        log.info("    measurement_%d  mean=%.1f mm  AUC@100mm=%.4f",
                 m, m_mm.mean(), compute_add_metrics(m_mm)["add_auc_100mm"])

    err_mm  = np.array(all_err_raw, dtype=np.float64) * 1000.0
    if len(err_mm) == 0:
        log.error("No frames evaluated - check dataset.data_folder paths.")
        return

    metrics = compute_add_metrics(err_mm)
    metrics.update(aggregate_frame_diagnostics(all_diags))
    metrics["num_frames"] = len(all_err_raw)

    robot_tag = cfg.robot.name
    log.info("=== Hydra (%s) evaluation results ===", robot_tag)
    log.info("  ADD AUC @100mm: %.4f", metrics["add_auc_100mm"])
    log.info("  ADD AUC @400mm: %.4f", metrics["add_auc_400mm"])
    log.info("  ADD @100mm:     %.4f", metrics["add_at_100mm"])
    log.info("  ADD @50mm:      %.4f", metrics["add_at_50mm"])
    log.info("  Mean ADD (mm):  %.2f  [median=%.2f  P90=%.2f  P95=%.2f]",
             metrics["add_mean_mm"], metrics["add_median_mm"],
             metrics["add_p90_mm"],  metrics["add_p95_mm"])
    log_diagnostics_summary(log, metrics)

    save_results = cfg.get("save_results", None)
    if save_results:
        save_eval_results(
            save_results, cfg, dataset=f"hydra_{robot_tag}", metrics=metrics,
            model_stats=model_stats, det_meta=det_meta, queries=all_queries,
        )


if __name__ == "__main__":
    main()
