"""Evaluate OWI-535 poses with the CRAVES lab-test-real protocol.

Reports PCK@0.2 for all 17 keypoints and the three joint keypoints, joint PCK
AUC, pose error and joint-position ADD.

"""
import logging
from typing import Optional, Tuple

# This import must precede torch and cv2 for deterministic environment settings.
from eval_utils import (
    seed_everything, build_eval_stack, compute_add_metrics,
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
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from loaders.craves_loader import (
    CRAVESLabDataset, craves_native_size,
    CRAVES_NUM_JOINT_KP, craves_keypoint_offsets, predict_all_17_keypoints_2d,
)

log = logging.getLogger(__name__)


# FK(joint_angles) returns [Model, Rotation, Base, Elbow, Wrist].  The CRAVES
# Evaluated joint keypoints are [Base, Elbow, Wrist] (index 1..3 of
# joint_name after dropping `Rotation`).  So in FK output that's links 2, 3, 4.
FK_JOINT_KP_INDICES = [2, 3, 4]

PCK_ALPHA = 0.2
PCK_ALPHA_SWEEP = np.arange(0.05, 0.505, 0.01)
# Derive the 0.2 key from the sweep because NumPy versions spell its float
# representation differently.
PCK_ALPHA_KEY = float(PCK_ALPHA_SWEEP[int(np.argmin(np.abs(PCK_ALPHA_SWEEP - PCK_ALPHA)))])


def _geodesic_rot_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_pred.T @ R_gt
    cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _euler_rot_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Mean absolute Euler-angle error (xyz, degrees) - matches RoboPose Table 3 'Rot.'"""
    R_err = R_pred.T @ R_gt
    angles = Rotation.from_matrix(R_err).as_euler("xyz", degrees=True)
    return float(np.mean(np.abs(angles)))


def _compute_frame_metrics(
    cTr_np: np.ndarray,
    T0C_gt: np.ndarray,
    joints_rad: torch.Tensor,
    kp2d_gt: np.ndarray,
    kp_offsets: np.ndarray,
    kp_link_idxs: list,
    bbox: np.ndarray,
    K_native: np.ndarray,
    robot_kin,
) -> dict:
    """
    Per-frame CRAVES metrics.  `cTr_np` is the predicted world->camera 4×4
    (world ≡ robot-base for OWI-535); `T0C_gt` is the GT world->camera.

    Pixel spaces: `kp2d_gt` and `bbox` are at native 1280×720.  Predicted 2D
    points are computed using `K_native` so both sets share the same pixel frame.
    """
    R_p, t_p = cTr_np[:3, :3], cTr_np[:3, 3]
    R_g, t_g = T0C_gt[:3, :3], T0C_gt[:3, 3]

    trans_err_cm = float(np.linalg.norm(t_p - t_g)) * 100.0
    trans_err_xyz_cm = float(np.mean(np.abs(t_p - t_g))) * 100.0
    rot_err_deg = _geodesic_rot_error_deg(R_p, R_g)
    rot_euler_deg = _euler_rot_error_deg(R_p, R_g)

    R_list, t_list = robot_kin.get_joint_R_t(joints_rad)
    t_list_np = np.asarray(t_list, dtype=np.float64)          # (5, 3) metres
    fk_kp = t_list_np[FK_JOINT_KP_INDICES]                    # (3, 3)

    pts_cam_pred = (R_p @ fk_kp.T).T + t_p                    # (3, 3) metres
    pts_cam_gt   = (R_g @ fk_kp.T).T + t_g                    # (3, 3) metres
    add_per_kp_mm = np.linalg.norm(pts_cam_pred - pts_cam_gt, axis=1) * 1000.0

    bbox_scale = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))

    R_list_np = np.asarray(R_list, dtype=np.float64)
    kp2d_pred_all17 = predict_all_17_keypoints_2d(
        kp_offsets, kp_link_idxs, R_list_np, t_list_np, R_p, t_p, K_native
    )                                                           # (17, 2)
    dist_all17 = np.linalg.norm(kp2d_pred_all17 - kp2d_gt, axis=1)  # (17,)
    pck_per_alpha_17 = {
        float(a): (dist_all17 < a * bbox_scale).astype(np.float32)
        for a in PCK_ALPHA_SWEEP
    }

    dist_joints = dist_all17[:CRAVES_NUM_JOINT_KP]
    pck_per_alpha = {
        float(a): (dist_joints < a * bbox_scale).astype(np.float32)
        for a in PCK_ALPHA_SWEEP
    }

    return {
        "trans_err_cm": trans_err_cm,
        "trans_err_xyz_cm": trans_err_xyz_cm,
        "rot_err_deg": rot_err_deg,
        "rot_euler_deg": rot_euler_deg,
        "add_per_kp_mm": add_per_kp_mm,                        # (3,)
        "pck_per_kp_alpha": pck_per_alpha,                     # dict α -> (3,) {0,1}
        "pck_per_kp_alpha_17": pck_per_alpha_17,               # dict α -> (17,) {0,1}
        "kp2d_pred": kp2d_pred_all17[:CRAVES_NUM_JOINT_KP],
        "kp2d_pred_all17": kp2d_pred_all17,                    # (17, 2)
        "bbox_scale": bbox_scale,
    }


def evaluate_craves(
    adapter,
    dataset: CRAVESLabDataset,
    kp_offsets: np.ndarray,
    kp_link_idxs: list,
    use_gpu: bool = True,
    return_per_frame: bool = False,
    detections: dict = None,
    detection_ids: set = None,
    num_eval_frames: Optional[int] = None,
) -> Tuple[dict, list]:
    queries = [] if return_per_frame else None
    trans_err_cm, trans_err_xyz_cm, rot_err_deg, rot_euler_deg = [], [], [], []
    per_frame_add_mm = []   # one mean-ADD value per frame; failures get 1000 mm sentinel
    pck_flat    = {float(a): [] for a in PCK_ALPHA_SWEEP}
    pck_flat_17 = {float(a): [] for a in PCK_ALPHA_SWEEP}
    n_fail = 0

    # Collected only on successful frames - used as a scale sanity check (ratio ~ 1).
    diag_t_pred_norms: list[float] = []
    diag_t_gt_norms: list[float] = []
    _all_diags: list = []

    robot_kin = adapter.estimator.robot_kin

    # Detection-aware subsetting: an int num_eval_frames is drawn from the frames the
    # detections file covers (a no-op for full-coverage files) - see
    # eval_utils.select_eval_frame_indices.
    _covered = None
    if detection_ids:
        _covered = [i for i, fid in enumerate(dataset.frame_ids) if fid in detection_ids] or None
    frame_indices = select_eval_frame_indices(len(dataset), num_eval_frames, _covered)
    _subset_src = f" of {len(_covered)} detection-covered frames" if _covered is not None else ""
    log.info(f"Evaluating {len(frame_indices)} of {len(dataset)} frames"
             + (f" (evenly-spaced subset{_subset_src})" if len(frame_indices) < len(dataset)
                else " (full dataset)"))

    verify_detection_coverage(detections or {}, detection_ids or set(),
                              [dataset.frame_ids[i] for i in frame_indices], "craves")

    for idx in tqdm(frame_indices):
        sample = dataset[idx]
        image = sample["image"]
        joints_rad = sample["joints_rad"]
        T0C_gt = sample["T0C"].astype(np.float64)
        kp2d_gt = sample["kp2d_gt"].astype(np.float64)
        bbox = sample["bbox"].astype(np.float64)
        frame_id = sample["frame_id"]
        K_native_np = sample["K_native"].astype(np.float64)

        if use_gpu and torch.cuda.is_available():
            image = image.cuda()

        # frame_id matches robot-detector's iter_craves output (filename stem, e.g. "00000037")
        cTr, est_diag = run_frame_inference(
            adapter, image, joints_rad.cpu().squeeze(), str(frame_id),
            detections, detection_ids,
            frame_key=str(dataset.img_dir / f"{frame_id}.jpg"),
            K=sample["K"],   # per-frame K from caminfo.json, never frame zero's
        )
        _all_diags.append(est_diag)

        if cTr is None:
            n_fail += 1
            trans_err_cm.append(np.nan)
            trans_err_xyz_cm.append(np.nan)
            rot_err_deg.append(np.nan)
            rot_euler_deg.append(np.nan)
            per_frame_add_mm.append(1000.0)
            for a in PCK_ALPHA_SWEEP:
                pck_flat[float(a)].extend([0.0] * CRAVES_NUM_JOINT_KP)
                pck_flat_17[float(a)].extend([0.0] * 17)
            if return_per_frame:
                H_img, W_img = int(image.shape[-2]), int(image.shape[-1])
                queries.append({
                    "frame_id":      frame_id,
                    "pnp_failed":    True,
                    "image_path":    str(dataset.img_dir / f"{frame_id}.jpg"),
                    "image_size_hw": [H_img, W_img],
                    "K_eval":        sample["K"].astype(float).tolist(),
                    "K_native":      K_native_np.tolist(),
                    "joints_rad":    joints_rad.cpu().numpy().tolist(),
                    "gt_pose":       T0C_gt.tolist(),
                    "est_pose":      None,
                    "add_m_mm":      1000.0,
                    "trans_err_cm":          None,
                    "trans_err_xyz_cm":      None,
                    "rot_err_deg":           None,
                    "rot_euler_deg":         None,
                    "add_per_kp_mm":         None,
                    "kp2d_pred_native":      None,
                    "kp2d_pred_all17_native": None,
                    "kp2d_gt_joints_native": kp2d_gt[:CRAVES_NUM_JOINT_KP].tolist(),
                    "kp2d_gt_all_native":    kp2d_gt.tolist(),
                    "bbox_native":   bbox.tolist(),
                    "bbox_scale":    None,
                    "t_pred_norm":   None,
                    "t_gt_norm":     float(np.linalg.norm(T0C_gt[:3, 3])),
                    "diagnostics":   est_diag,
                })
            continue

        cTr_np = _to_np(cTr).astype(np.float64).squeeze()

        frame = _compute_frame_metrics(
            cTr_np, T0C_gt, joints_rad, kp2d_gt, kp_offsets, kp_link_idxs, bbox, K_native_np, robot_kin,
        )

        # First-valid-frame guard: both translations are the base origin in the camera
        # frame (the loader normalises CRAVES GT to metres, UE units × 0.001), so they
        # must agree in unit. Catches a silent mesh-mm vs GT-metres mismatch (~1000×).
        # The mean-ratio diagnostic below stays: it reports the finer scale bias that
        # this hard [0.01, 100] band deliberately does not police.
        assert_translation_units_consistent(cTr_np[:3, 3], T0C_gt[:3, 3], tag="craves")

        diag_t_pred_norms.append(float(np.linalg.norm(cTr_np[:3, 3])))
        diag_t_gt_norms.append(float(np.linalg.norm(T0C_gt[:3, 3])))

        trans_err_cm.append(frame["trans_err_cm"])
        trans_err_xyz_cm.append(frame["trans_err_xyz_cm"])
        rot_err_deg.append(frame["rot_err_deg"])
        rot_euler_deg.append(frame["rot_euler_deg"])
        per_frame_add_mm.append(float(np.mean(frame["add_per_kp_mm"])))
        for a, arr in frame["pck_per_kp_alpha"].items():
            pck_flat[a].extend(arr.tolist())
        for a, arr in frame["pck_per_kp_alpha_17"].items():
            pck_flat_17[a].extend(arr.tolist())

        if return_per_frame:
            H_img, W_img = int(image.shape[-2]), int(image.shape[-1])
            queries.append({
                "frame_id":   frame_id,
                "pnp_failed": False,
                "image_path": str(dataset.img_dir / f"{frame_id}.jpg"),
                "image_size_hw": [H_img, W_img],
                "K_eval":     sample["K"].astype(float).tolist(),
                "K_native":   K_native_np.tolist(),
                "joints_rad": joints_rad.cpu().numpy().tolist(),
                "gt_pose":    T0C_gt.tolist(),
                "est_pose":   cTr_np.tolist(),
                "add_m_mm":   round(float(np.mean(frame["add_per_kp_mm"])), 3),
                "trans_err_cm":          frame["trans_err_cm"],
                "trans_err_xyz_cm":      frame["trans_err_xyz_cm"],
                "rot_err_deg":           frame["rot_err_deg"],
                "rot_euler_deg":         frame["rot_euler_deg"],
                "add_per_kp_mm":         frame["add_per_kp_mm"].tolist(),
                "kp2d_pred_native":      frame["kp2d_pred"].tolist(),
                "kp2d_pred_all17_native": frame["kp2d_pred_all17"].tolist(),
                "kp2d_gt_joints_native": kp2d_gt[:CRAVES_NUM_JOINT_KP].tolist(),
                "kp2d_gt_all_native":    kp2d_gt.tolist(),
                "bbox_native":  bbox.tolist(),
                "bbox_scale":   frame["bbox_scale"],
                "t_pred_norm":  float(np.linalg.norm(cTr_np[:3, 3])),
                "t_gt_norm":    float(np.linalg.norm(T0C_gt[:3, 3])),
                "diagnostics":  est_diag,
            })

    pck_at_0_2 = float(np.mean(pck_flat[PCK_ALPHA_KEY])) if pck_flat[PCK_ALPHA_KEY] else 0.0
    pck_auc = float(np.mean([np.mean(v) if v else 0.0 for v in pck_flat.values()]))
    pck_at_0_2_17 = float(np.mean(pck_flat_17[PCK_ALPHA_KEY])) if pck_flat_17[PCK_ALPHA_KEY] else 0.0
    pck_auc_17 = float(np.mean([np.mean(v) if v else 0.0 for v in pck_flat_17.values()]))
    trans_err_cm_arr = np.asarray(trans_err_cm, dtype=np.float64)
    trans_err_xyz_cm_arr = np.asarray(trans_err_xyz_cm, dtype=np.float64)
    rot_err_deg_arr = np.asarray(rot_err_deg, dtype=np.float64)
    rot_euler_deg_arr = np.asarray(rot_euler_deg, dtype=np.float64)

    if diag_t_pred_norms and diag_t_gt_norms:
        ratio = np.array(diag_t_pred_norms) / (np.array(diag_t_gt_norms) + 1e-9)
        diag_t_pred_mean = float(np.mean(diag_t_pred_norms))
        diag_t_gt_mean = float(np.mean(diag_t_gt_norms))
        diag_ratio_mean = float(np.mean(ratio))
    else:
        diag_t_pred_mean = diag_t_gt_mean = diag_ratio_mean = float("nan")

    add_metrics = compute_add_metrics(np.array(per_frame_add_mm)) if per_frame_add_mm else {}
    diag_agg = aggregate_frame_diagnostics(_all_diags)

    metrics = {
        "num_frames": len(frame_indices),
        "num_pnp_failures": n_fail,
        "pck_at_0_2_all17": pck_at_0_2_17,
        "pck_auc_all17": pck_auc_17,
        "pck_at_0_2_joints": pck_at_0_2,
        "pck_auc_joints": pck_auc,
        "mean_trans_err_cm": float(np.nanmean(trans_err_cm_arr)) if trans_err_cm else float("nan"),
        "median_trans_err_cm": float(np.nanmedian(trans_err_cm_arr)) if trans_err_cm else float("nan"),
        "mean_trans_err_xyz_cm": float(np.nanmean(trans_err_xyz_cm_arr)) if trans_err_xyz_cm else float("nan"),
        "median_trans_err_xyz_cm": float(np.nanmedian(trans_err_xyz_cm_arr)) if trans_err_xyz_cm else float("nan"),
        "mean_rot_err_deg": float(np.nanmean(rot_err_deg_arr)) if rot_err_deg else float("nan"),
        "median_rot_err_deg": float(np.nanmedian(rot_err_deg_arr)) if rot_err_deg else float("nan"),
        "mean_rot_euler_deg": float(np.nanmean(rot_euler_deg_arr)) if rot_euler_deg else float("nan"),
        "median_rot_euler_deg": float(np.nanmedian(rot_euler_deg_arr)) if rot_euler_deg else float("nan"),
        **add_metrics,
        "diag_mean_t_pred_norm_m": diag_t_pred_mean,
        "diag_mean_t_gt_norm_m": diag_t_gt_mean,
        "diag_mean_t_pred_over_t_gt": diag_ratio_mean,
    }
    metrics.update(diag_agg)
    return metrics, queries


@hydra.main(config_path="../configs", config_name="eval_craves", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    seed_everything(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scale = float(cfg.dataset.get("scale", 1.0))
    frame_slice = cfg.dataset.get("frame_slice", None)
    dataset = CRAVESLabDataset(
        data_folder=cfg.dataset.data_folder,
        scale=scale,
        frame_slice=tuple(frame_slice) if frame_slice is not None else None,
        trans_to_tensor=transforms.ToTensor(),
    )

    native_w, native_h = craves_native_size()
    sample0 = dataset[0]
    K_eval_np = sample0["K"].astype(np.float32)
    K_native_np = sample0["K_native"].astype(np.float32)
    log.info(
        f"K_native[0] = fx={K_native_np[0,0]:.1f} fy={K_native_np[1,1]:.1f} "
        f"cx={K_native_np[0,2]:.1f} cy={K_native_np[1,2]:.1f}  "
        f"native size={native_w}x{native_h}, scale={scale}"
    )

    eval_h = int(round(native_h * scale))
    eval_w = int(round(native_w * scale))

    adapter, model_stats = build_eval_stack(cfg, device, K_eval_np, (eval_h, eval_w))

    log.info(f"Dataset: {len(dataset)} CRAVES frames from {cfg.dataset.data_folder}")
    log.info("URDF<->UE bridge: canonical (R_URDF_TO_UE = diag(1,-1,1))")

    kp_offsets, kp_link_idxs = craves_keypoint_offsets()

    # Sanity: apply GT pose + FK to the keypoint offsets on one frame;
    # result should match GT 2D keypoints from make_2d_keypoints (~0 px).
    _s0 = sample0
    _q0 = _s0["joints_rad"].cpu().numpy()
    _R0, _t0 = adapter.estimator.robot_kin.get_joint_R_t(_q0)
    _R0g, _t0g = _s0["T0C"][:3, :3].astype(np.float64), _s0["T0C"][:3, 3].astype(np.float64)
    _K0 = _s0["K_native"].astype(np.float64)
    _pred_gt = predict_all_17_keypoints_2d(
        kp_offsets, kp_link_idxs, np.asarray(_R0), np.asarray(_t0), _R0g, _t0g, _K0
    )
    _gt_kp = _s0["kp2d_gt"].astype(np.float64)
    _err_px = float(np.mean(np.linalg.norm(_pred_gt - _gt_kp, axis=1)))
    log.info(
        f"[sanity] 17-kp offsets -> GT pose reprojection error on frame 0: {_err_px:.2f} px "
        f"(expect <5 px if FK frames match RoboPose)"
    )
    if _err_px > 20:
        log.warning(
            "Reprojection error > 20 px - CRAVES_KEYPOINT_OFFSETS may be "
            "in a different coordinate frame than the robot FK. Check the URDF."
        )

    detections, detection_ids, det_meta = load_detections_for_run(cfg)

    save_results = cfg.get("save_results", None)
    metrics, queries = evaluate_craves(
        adapter,
        dataset,
        kp_offsets=kp_offsets,
        kp_link_idxs=kp_link_idxs,
        use_gpu=torch.cuda.is_available(),
        return_per_frame=save_results is not None,
        detections=detections,
        detection_ids=detection_ids,
        num_eval_frames=cfg.get("num_eval_frames", None),
    )

    log.info("=== CRAVES evaluation results ===")
    log.info(f"  Num frames            : {metrics['num_frames']}")
    log.info(f"  PnP failures          : {metrics['num_pnp_failures']}")
    log.info(f"  PCK@0.2 (17 kp)       : {metrics['pck_at_0_2_all17']:.4f}  [RoboPose: 0.9920]")
    log.info(f"  PCK-AUC (17 kp)       : {metrics['pck_auc_all17']:.4f}")
    log.info(f"  PCK@0.2 (3 joints)    : {metrics['pck_at_0_2_joints']:.4f}")
    log.info(f"  PCK-AUC (3 joints)    : {metrics['pck_auc_joints']:.4f}")
    log.info("  --- translation (RoboPose synt targets: xyz=0.61cm  norm=1.31cm) ---")
    log.info(f"  Mean trans xyz (cm)   : {metrics['mean_trans_err_xyz_cm']:.2f}  [= RoboPose 'Trans xyz.']")
    log.info(f"  Mean trans norm (cm)  : {metrics['mean_trans_err_cm']:.2f}  [= RoboPose 'Trans norm.']")
    log.info("  --- rotation (RoboPose synt target: Euler=4.12 deg) ---")
    log.info(f"  Mean rot Euler (deg)  : {metrics['mean_rot_euler_deg']:.2f}  [= RoboPose 'Rot.']")
    log.info(f"  Mean rot geodesic     : {metrics['mean_rot_err_deg']:.2f}  (internal metric)")
    log.info(f"  ADD AUC @100mm        : {metrics.get('add_auc_100mm', float('nan')):.4f}")
    log.info(f"  ADD AUC @400mm        : {metrics.get('add_auc_400mm', float('nan')):.4f}")
    log.info(f"  ADD @100mm            : {metrics.get('add_at_100mm', float('nan')):.4f}")
    log.info(f"  ADD @50mm             : {metrics.get('add_at_50mm', float('nan')):.4f}")
    log.info(f"  Mean joint-ADD (mm)   : {metrics.get('add_mean_mm', float('nan')):.2f}  "
             f"[median={metrics.get('add_median_mm', float('nan')):.2f}  "
             f"P90={metrics.get('add_p90_mm', float('nan')):.2f}]")
    log_diagnostics_summary(log, metrics)
    log.info("  --- unit-sanity diagnostics (expect |t_pred|/|t_gt| ~ 1) ---")
    log.info(f"  mean ||t_pred||       : {metrics['diag_mean_t_pred_norm_m']:.3f} m")
    log.info(f"  mean ||t_gt||         : {metrics['diag_mean_t_gt_norm_m']:.3f} m")
    log.info(f"  mean ratio            : {metrics['diag_mean_t_pred_over_t_gt']:.3f}")

    if save_results:
        save_eval_results(
            save_results, cfg, dataset="craves", metrics=metrics,
            model_stats=model_stats, det_meta=det_meta, queries=queries,
        )


if __name__ == "__main__":
    main()
