import logging
import os
from pathlib import Path
from typing import List, Optional

# eval_utils sets CUBLAS_WORKSPACE_CONFIG / OMP_NUM_THREADS at import time and must
# therefore be imported before torch and cv2 - hence this import above the
# third-party block. See the eval_utils module docstring.
from eval_utils import (
    seed_everything, build_eval_stack, compute_add_metrics, project_points,
    aggregate_frame_diagnostics, log_diagnostics_summary, _to_np,
    load_detections_for_run, run_frame_inference, save_eval_results,
    select_eval_frame_indices, verify_detection_coverage,
    assert_translation_units_consistent,
)

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from loaders.dream_loader import DREAMDataset, load_camera_parameters

log = logging.getLogger(__name__)


# Geometry helpers

def _batch_transform_3d(T_4x4: torch.Tensor, pts3d) -> torch.Tensor:
    """Apply a (4, 4) pose to (N, 3) points -> (N, 3) in camera frame."""
    if isinstance(pts3d, np.ndarray):
        pts3d = torch.from_numpy(pts3d).float()
    pts3d = pts3d.to(T_4x4.device)
    R, t = T_4x4[:3, :3], T_4x4[:3, 3]
    return (pts3d @ R.T) + t


def _rigid_transform_3d(X, Y):
    """Least-squares rigid transform: X (base frame) -> Y (camera frame)."""
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    X_mean, Y_mean = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    H = (X - X_mean).T @ (Y - Y_mean)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2] *= -1
        R = Vt.T @ U.T
    t = (Y_mean.T - R @ X_mean.T).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _save_overlay(image_tensor, pts_gt_2d, pts_pred_2d, out_path):
    img = image_tensor.detach().cpu()
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img[:3].permute(1, 2, 0).numpy()
    else:
        img = img.numpy()
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)
    overlay = img.copy()
    for pt in pts_gt_2d:
        cv2.circle(overlay, (int(round(pt[0])), int(round(pt[1]))), 4, (0, 255, 0), -1, cv2.LINE_AA)
    for pt in pts_pred_2d:
        cv2.circle(overlay, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 255), -1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# Evaluation loop

def evaluate_ADD(
    model,
    dataset,
    keypoint_indices: List[int],
    use_gpu: bool = True,
    num_overlays: int = 5,
    return_per_frame: bool = False,
    K_full: Optional[np.ndarray] = None,
    detections: dict = None,   # image_id -> detection entry from load_robot_detections
    detection_ids: set = None,  # every frame_id present in the detections file
    num_eval_frames: Optional[int] = None,
    frame_shard: Optional[str] = None,
):
    """
    ADD metric over all (or an evenly-spaced subset of) frames.

    model   : EstimatorAdapter - has .estimator.robot_kin and .inference_single_image.
    dataset : DREAMDataset - has .get_data_with_keypoints(i).
    num_eval_frames : None = evaluate every frame (final numbers). An int < len(dataset)
        evaluates that many evenly-spaced frames instead - see eval_utils.select_eval_frame_indices.

    Returns (metrics dict, queries list or None).  queries is None when
    return_per_frame=False; otherwise a list of per-frame dicts matching the
    structure written by the other eval scripts.
    """
    err_3d_list: list = []
    queries = [] if return_per_frame else None
    _all_diags: list = []
    count_overlays = 0

    _all_frame_ids = [Path(dataset._ndds_list[i]["image_paths"]["rgb"]).name.split(".")[0]
                      for i in range(len(dataset))]
    # Detection-aware subsetting: with a subsampled detections file (panda-orb's
    # covers only 1000 of the ~32k frames), an int num_eval_frames is drawn from the
    # covered frames so any N works - see eval_utils.select_eval_frame_indices.
    _covered = None
    if detection_ids:
        _covered = [i for i in range(len(dataset)) if _all_frame_ids[i] in detection_ids] or None
    frame_indices = select_eval_frame_indices(len(dataset), num_eval_frames, _covered, frame_shard)
    _subsampled = num_eval_frames is not None and num_eval_frames < len(dataset)
    _subset_src = f" of {len(_covered)} detection-covered frames" if _covered is not None else ""
    _note = f" (evenly-spaced subset{_subset_src})" if _subsampled else " (full dataset)"
    if frame_shard is not None:
        _note += f", shard {frame_shard}"
    log.info(f"Evaluating {len(frame_indices)} of {len(dataset)} frames{_note}")

    _sel_ids = [_all_frame_ids[i] for i in frame_indices]
    verify_detection_coverage(detections or {}, detection_ids or set(), _sel_ids, "panda_orb")

    for data_n in tqdm(frame_indices):
        img, joint_angles, keypoints = dataset.get_data_with_keypoints(data_n)
        if use_gpu:
            img = img.cuda()

        joints_cpu = joint_angles.cpu().squeeze()

        img_path = dataset._ndds_list[data_n]["image_paths"]["rgb"]
        # Digit-prefixed image stem (e.g. "000042"), robust to iteration-order
        # differences between robot-detector and DREAMDataset. Stored in every
        # query so panda_orb results are frame-keyable like the other datasets.
        frame_id = Path(img_path).name.split(".")[0]
        cTr, _frame_diag = run_frame_inference(
            model, img, joints_cpu, frame_id, detections, detection_ids,
            frame_key=img_path,
        )
        _all_diags.append(_frame_diag)
        kp_cam_gt = np.array([kp["location"] for kp in keypoints], dtype=np.float32)

        if cTr is None:
            log.debug(f"Frame {data_n:04d}: PnP failed, assigning max error.")
            err_3d_list.append(1.0)
            if return_per_frame:
                _, t_list = model.estimator.robot_kin.get_joint_R_t(joints_cpu)
                X_base = np.asarray(t_list, dtype=np.float32)[keypoint_indices]
                gt_T = _rigid_transform_3d(X_base, kp_cam_gt)
                queries.append({
                    "image_path": dataset._ndds_list[data_n]["image_paths"]["rgb"],
                    "frame_id": frame_id,
                    "K_eval": K_full.tolist() if K_full is not None else None,
                    "image_size_hw": [img.shape[-2], img.shape[-1]],
                    "gt_pose": gt_T.tolist(),
                    "est_pose": None,
                    "pnp_failed": True,
                    "add_m_mm": 1000.0,
                    "keypoints_base_m": X_base.tolist(),
                    "keypoints_cam_gt": kp_cam_gt.tolist(),
                    "joints_rad": joints_cpu.numpy().tolist(),
                    "add_m": 1.0,
                    "diagnostics": _frame_diag,
                })
            continue

        _, t_list = model.estimator.robot_kin.get_joint_R_t(joints_cpu)
        points_3d = torch.from_numpy(np.asarray(t_list, dtype=np.float32))
        if use_gpu:
            points_3d = points_3d.cuda()

        points_3d_pred = _to_np(_batch_transform_3d(cTr, points_3d)).squeeze()
        points_3d_pred = points_3d_pred[keypoint_indices]

        err_3d = np.mean(np.linalg.norm(points_3d_pred - kp_cam_gt, axis=1))
        err_3d_list.append(err_3d)

        # GT pose from Kabsch on the GT keypoints (also recorded per frame below).
        X_base = np.asarray(t_list, dtype=np.float32)[keypoint_indices]
        gt_T = _rigid_transform_3d(X_base, kp_cam_gt)
        # First-valid-frame guard: predicted base translation and GT base translation are
        # both the base origin in the camera frame, so they must agree in unit. Catches a
        # silent mesh-mm vs GT-metres mismatch (~1000×).
        assert_translation_units_consistent(
            _to_np(cTr).squeeze()[:3, 3], gt_T[:3, 3], tag="panda_orb")

        if count_overlays < num_overlays:
            K_np = model._K.detach().cpu().numpy()
            points_2d_gt = np.array([kp["projected_location"] for kp in keypoints])
            _save_overlay(
                img, points_2d_gt, project_points(K_np, points_3d_pred),
                os.path.join("debug_templates", f"overlay_{data_n:04d}.png"),
            )
            count_overlays += 1

        if return_per_frame:
            queries.append({
                "image_path": dataset._ndds_list[data_n]["image_paths"]["rgb"],
                "frame_id": frame_id,
                "K_eval": K_full.tolist() if K_full is not None else None,
                "image_size_hw": [img.shape[-2], img.shape[-1]],
                "gt_pose": gt_T.tolist(),
                "est_pose": np.asarray(_to_np(cTr), dtype=np.float32).tolist(),
                "pnp_failed": False,
                "add_m_mm": round(float(err_3d) * 1000, 3),
                "keypoints_base_m": X_base.tolist(),
                "keypoints_cam_gt": kp_cam_gt.tolist(),
                "joints_rad": joints_cpu.numpy().tolist(),
                "add_m": float(err_3d),
                "diagnostics": _frame_diag,
            })

    err_3d_mm = np.array(err_3d_list).flatten() * 1000.0
    metrics = compute_add_metrics(err_3d_mm)
    metrics.update(aggregate_frame_diagnostics(_all_diags))

    log.info(
        f"ADD evaluation complete - "
        f"AUC@100mm: {metrics['add_auc_100mm']:.4f}, "
        f"AUC@400mm: {metrics['add_auc_400mm']:.4f}, "
        f"mean: {metrics['add_mean_mm']:.2f} mm"
        + (f", runtime median: {metrics['runtime_median_ms']:.1f} ms" if "runtime_median_ms" in metrics else "")
    )

    return metrics, queries


# Entry point

@hydra.main(config_path="../configs", config_name="eval_panda_orb", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fx, fy, cx, cy = load_camera_parameters(cfg.dataset.data_folder)
    height = cfg.robot.image_height
    width = cfg.robot.image_width

    K_full = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    adapter, model_stats = build_eval_stack(cfg, device, K_full, (height, width))

    dataset = DREAMDataset(
        data_folder=cfg.dataset.data_folder,
        trans_to_tensor=transforms.Compose([transforms.ToTensor()]),
    )

    keypoint_indices = list(cfg.robot.eval_keypoint_indices)
    log.info(f"Dataset: {len(dataset)} frames from {cfg.dataset.data_folder}")
    log.info(f"Keypoint indices for ADD: {keypoint_indices}")

    detections, detection_ids, det_meta = load_detections_for_run(cfg)

    save_results = cfg.get("save_results", None)
    metrics, queries = evaluate_ADD(
        adapter, dataset,
        use_gpu=torch.cuda.is_available(),
        num_overlays=cfg.num_overlays,
        keypoint_indices=keypoint_indices,
        return_per_frame=save_results is not None,
        K_full=K_full,
        detections=detections,
        detection_ids=detection_ids,
        num_eval_frames=cfg.get("num_eval_frames", None),
        frame_shard=cfg.get("frame_shard", None),
    )

    log.info("=== Panda-ORB evaluation results ===")
    log.info(f"  ADD AUC @100mm: {metrics['add_auc_100mm']:.4f}")
    log.info(f"  ADD AUC @400mm: {metrics['add_auc_400mm']:.4f}")
    log.info(f"  ADD @100mm:     {metrics['add_at_100mm']:.4f}")
    log.info(f"  ADD @50mm:      {metrics['add_at_50mm']:.4f}")
    log.info(f"  Mean ADD (mm):  {metrics['add_mean_mm']:.2f}  "
             f"[median={metrics['add_median_mm']:.2f}  P90={metrics['add_p90_mm']:.2f}  "
             f"P95={metrics['add_p95_mm']:.2f}]")
    log_diagnostics_summary(log, metrics)

    if save_results:
        save_eval_results(
            save_results, cfg, dataset="panda_orb", metrics=metrics,
            extra_summary={"num_samples": len(queries)},
            model_stats=model_stats, det_meta=det_meta, queries=queries,
        )


if __name__ == "__main__":
    main()
