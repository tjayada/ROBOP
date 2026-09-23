"""Evaluate Baxter endpoint error with the CtRNet protocol.

Reports 2D PCK AUC over 0-200 px and 3D ADD AUC over 0-400 mm, including
the 50 px and 100 mm thresholds.
"""
import logging

# This import must precede torch and cv2 for deterministic environment settings.
from eval_utils import (
    seed_everything, build_eval_stack, compute_add_metrics, project_points,
    aggregate_frame_diagnostics, log_diagnostics_summary,
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

from loaders.baxter_loader import (
    BaxterDataset, baxter_K_full, baxter_K_scaled, BAXTER_EVAL_SCALE,
    BAXTER_WIDTH_FULL, BAXTER_HEIGHT_FULL,
)

log = logging.getLogger(__name__)


def _T_from_DH(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """CtRNet-style DH transform."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0, a],
        [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
        [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha),  np.cos(alpha),  d * np.cos(alpha)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _baxter_ee_base_ctrnet(joints: np.ndarray) -> np.ndarray:
    """
    Compute Baxter EE position in base frame exactly like CtRNet's get_bl_T_Jn(8, theta).
    """
    q = np.asarray(joints, dtype=np.float32).reshape(-1)
    if q.shape[0] != 7:
        raise ValueError(f"Expected 7 Baxter joints, got {q.shape[0]}")

    bl_T_0 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.27035],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    T_7_ee = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3683],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)

    T_0_1 = _T_from_DH(0.0,        0.0,   0.0,     q[0])
    T_1_2 = _T_from_DH(-np.pi/2.0, 0.069, 0.0,     q[1] + np.pi/2.0)
    T_2_3 = _T_from_DH(np.pi/2.0,  0.0,   0.36435, q[2])
    T_3_4 = _T_from_DH(-np.pi/2.0, 0.069, 0.0,     q[3])
    T_4_5 = _T_from_DH(np.pi/2.0,  0.0,   0.37429, q[4])
    T_5_6 = _T_from_DH(-np.pi/2.0, 0.010, 0.0,     q[5])
    T_6_7 = _T_from_DH(np.pi/2.0,  0.0,   0.0,     q[6])

    T = bl_T_0 @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ T_7_ee
    return T[:3, 3].astype(np.float32)


def evaluate_baxter(model, dataset, K_full_np, K_eval_np, use_gpu=True,
                    return_per_frame=False, detections=None, detection_ids=None,
                    num_eval_frames=None):
    """Evaluate Baxter endpoint PCK and ADD, optionally returning frame records."""
    err_2d_list = []
    err_3d_list = []
    _all_diags: list = []
    queries = [] if return_per_frame else None

    # Precompute frame_id for every sample to match robot-detector's format:
    #   frame_id = f"{pose_key}_{img_idx_within_pose:04d}"
    # The robot-detector iterates images sorted within each pose; _samples is built
    # in the same sorted order, so a per-pose counter gives the correct img_idx.
    _pose_counters: dict = {}
    _frame_ids: list = []
    for _img_path, _pk in dataset._samples:
        _c = _pose_counters.get(_pk, 0)
        _frame_ids.append(f"{_pk}_{_c:04d}")
        _pose_counters[_pk] = _c + 1

    # Detection-aware subsetting: an int num_eval_frames is drawn from the frames the
    # detections file covers (a no-op for full-coverage files) - see
    # eval_utils.select_eval_frame_indices.
    _covered = None
    if detection_ids:
        _covered = [i for i, fid in enumerate(_frame_ids) if fid in detection_ids] or None
    frame_indices = select_eval_frame_indices(len(dataset), num_eval_frames, _covered)
    _subset_src = f" of {len(_covered)} detection-covered frames" if _covered is not None else ""
    log.info(f"Evaluating {len(frame_indices)} of {len(dataset)} frames"
             + (f" (evenly-spaced subset{_subset_src})" if len(frame_indices) < len(dataset)
                else " (full dataset)"))

    verify_detection_coverage(detections or {}, detection_ids or set(),
                              [_frame_ids[i] for i in frame_indices], "baxter")

    for idx in tqdm(frame_indices):
        image, joint_angles, ee_2d_gt, ee_3d_gt, pose_key = dataset[idx]
        img_path = dataset._samples[idx][0]
        log.debug("Frame %04d (%s) from %s", idx, pose_key, img_path)
        if use_gpu and torch.cuda.is_available():
            image = image.cuda()

        joints_cpu = joint_angles.cpu().squeeze()

        cTr, diag = run_frame_inference(
            model, image, joints_cpu, _frame_ids[idx],   # e.g. "pose_3_0042"
            detections, detection_ids, frame_key=img_path,
        )
        _all_diags.append(diag)

        if cTr is None:
            log.debug(f"Frame {idx:04d} ({pose_key}): PnP failed, assigning max error.")
            err_2d_list.append(200.0)
            err_3d_list.append(1.0)
            if return_per_frame:
                queries.append({
                    "image_path": img_path,
                    "pose_key":   pose_key,
                    "frame_id":   _frame_ids[idx],
                    "image_size_hw": [image.shape[-2], image.shape[-1]],
                    # ee_2d_* are projected with K_full at native resolution;
                    # the viewer scales them by image_size_hw / native_size_hw.
                    "native_size_hw": [BAXTER_HEIGHT_FULL, BAXTER_WIDTH_FULL],
                    "joints_rad": joints_cpu.numpy().tolist(),
                    "K_eval":     K_eval_np.tolist(),
                    "ee_3d_gt":   ee_3d_gt.tolist(),
                    "ee_2d_gt_full_res": ee_2d_gt.tolist(),
                    "est_pose":    None,
                    "ee_cam_pred": None,
                    "ee_2d_pred":  None,
                    "err_2d_px":   200.0,
                    "err_3d_m":    1.0,
                    "add_m_mm":    1000.0,
                    "pnp_failed":  True,
                    "diagnostics": diag,
                })
            continue

        # Compute EE in the base frame with CtRNet's exact DH chain, avoiding
        # URDF ambiguity at the wrist.
        ee_base = _baxter_ee_base_ctrnet(joints_cpu.numpy())

        cTr_np = cTr.detach().cpu().numpy() if isinstance(cTr, torch.Tensor) else np.asarray(cTr)
        R_c, t_c = cTr_np[:3, :3], cTr_np[:3, 3]
        ee_cam_pred = R_c @ ee_base + t_c

        # First-valid-frame guard. Baxter has no GT base pose, so the comparable pair is
        # the end-effector point in the camera frame: predicted vs GT. A silent mesh-mm
        # vs GT-metres mismatch shows up there as the same ~1000× discrepancy.
        assert_translation_units_consistent(ee_cam_pred, ee_3d_gt, tag="baxter")

        err_3d = float(np.linalg.norm(ee_cam_pred - ee_3d_gt))
        err_3d_list.append(err_3d)

        ee_2d_pred = project_points(K_full_np, ee_cam_pred[None])[0]
        err_2d = float(np.linalg.norm(ee_2d_pred - ee_2d_gt))
        err_2d_list.append(err_2d)

        if return_per_frame:
            queries.append({
                "image_path": img_path,
                "pose_key":   pose_key,
                "frame_id":   _frame_ids[idx],
                "image_size_hw": [image.shape[-2], image.shape[-1]],
                # ee_2d_* are projected with K_full at native resolution;
                # the viewer scales them by image_size_hw / native_size_hw.
                "native_size_hw": [BAXTER_HEIGHT_FULL, BAXTER_WIDTH_FULL],
                "joints_rad": joints_cpu.numpy().tolist(),
                "K_eval":     K_eval_np.tolist(),
                "ee_3d_gt":   ee_3d_gt.tolist(),
                "ee_2d_gt_full_res": ee_2d_gt.tolist(),
                "est_pose":    cTr_np.tolist(),
                "ee_cam_pred": ee_cam_pred.tolist(),
                "ee_2d_pred":  ee_2d_pred.tolist(),
                "err_2d_px":   err_2d,
                "err_3d_m":    err_3d,
                "add_m_mm":    round(err_3d * 1000, 3),
                "pnp_failed":  False,
                "diagnostics": diag,
            })

    err_2d_arr = np.array(err_2d_list, dtype=np.float64)
    err_3d_mm  = np.array(err_3d_list, dtype=np.float64) * 1000.0

    pck_auc   = float(np.mean(err_2d_arr[:, None] < np.arange(200, dtype=np.float64)))
    pck_at_50 = float(np.mean(err_2d_arr < 50))

    add_metrics = compute_add_metrics(err_3d_mm)

    metrics = {
        "mean_2d_px":    float(np.mean(err_2d_arr)),
        "pck_auc_200px": pck_auc,
        "pck_at_50px":   pck_at_50,
        **add_metrics,
        "num_frames":    len(err_2d_list),
    }
    metrics.update(aggregate_frame_diagnostics(_all_diags))
    return metrics, queries


@hydra.main(config_path="../configs", config_name="eval_baxter", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = cfg.dataset.get("scale", BAXTER_EVAL_SCALE)
    K_full_np = baxter_K_full()
    K_eval_np = baxter_K_scaled(scale)

    height = int(cfg.robot.image_height * scale)
    width  = int(cfg.robot.image_width  * scale)

    adapter, model_stats = build_eval_stack(cfg, device, K_eval_np, (height, width))

    dataset = BaxterDataset(
        data_folder=cfg.dataset.data_folder,
        scale=scale,
        trans_to_tensor=transforms.ToTensor(),
    )
    log.info(f"Dataset: {len(dataset)} frames from {cfg.dataset.data_folder}")

    detections, detection_ids, det_meta = load_detections_for_run(cfg)

    save_results = cfg.get("save_results", None)
    metrics, queries = evaluate_baxter(
        adapter,
        dataset,
        K_full_np=K_full_np,
        K_eval_np=K_eval_np,
        use_gpu=torch.cuda.is_available(),
        return_per_frame=save_results is not None,
        detections=detections,
        detection_ids=detection_ids,
        num_eval_frames=cfg.get("num_eval_frames", None),
    )

    log.info("=== Baxter evaluation results ===")
    log.info(f"  Mean 2D EE error : {metrics['mean_2d_px']:.2f} px")
    log.info(f"  PCK AUC (0-200px): {metrics['pck_auc_200px']:.4f}")
    log.info(f"  PCK @50 px       : {metrics['pck_at_50px']:.4f}")
    log.info(f"  Mean 3D EE (mm)  : {metrics['add_mean_mm']:.2f}  "
             f"[median={metrics['add_median_mm']:.2f}  P90={metrics['add_p90_mm']:.2f}]")
    log.info(f"  ADD AUC @100mm   : {metrics['add_auc_100mm']:.4f}")
    log.info(f"  ADD AUC @400mm   : {metrics['add_auc_400mm']:.4f}")
    log.info(f"  ADD @100mm       : {metrics['add_at_100mm']:.4f}")
    log.info(f"  ADD @50mm        : {metrics['add_at_50mm']:.4f}")
    log_diagnostics_summary(log, metrics)

    if save_results:
        save_eval_results(
            save_results, cfg, dataset="baxter", metrics=metrics,
            model_stats=model_stats, det_meta=det_meta, queries=queries,
        )


if __name__ == "__main__":
    main()
