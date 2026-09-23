"""Score bundled RoboPose CRAVES predictions with the joint-ADD metric.

Predicted camera-space Base, Elbow and Wrist positions are compared with the
ground-truth pose and FK keypoints used by ``evaluation/run_eval_craves.py``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_eval_on_path() -> None:
    """Make the evaluation package importable from this script directory."""
    eval_dir = str(Path(__file__).resolve().parents[2] / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)


def compute_add_metrics(per_frame_add_mm: np.ndarray) -> dict:
    """Use the same DREAM/CtRNet ADD-AUC implementation as evaluation runs."""
    _ensure_eval_on_path()
    from eval_utils import compute_add_metrics as _canonical
    return _canonical(np.asarray(per_frame_add_mm, dtype=np.float64))


FK_JOINT_KP_INDICES = [2, 3, 4]   # Base, Elbow, Wrist in FK output (5 links total)


def load_craves_gt(data_folder: str, scene_ids: np.ndarray):
    """
    Returns aligned lists of T0C_gt (4x4) and joints_rad (4,) for each
    scene_id in scene_ids, plus the robot_kin object.
    """
    try:
        from loaders.craves_loader import CRAVESLabDataset
    except ImportError:
        # The loaders package uses script-dir imports ("from loaders.utils import ..."),
        # so evaluation/ itself has to be on sys.path for craves_loader to resolve.
        _ensure_eval_on_path()
        from loaders.craves_loader import CRAVESLabDataset

    dataset = CRAVESLabDataset(data_folder=data_folder, scale=1.0)
    id_to_idx = {fid: i for i, fid in enumerate(dataset.frame_ids)}

    T0C_list, joints_list, found_ids = [], [], []
    missing = []
    for sid in scene_ids:
        if sid in id_to_idx:
            sample = dataset[id_to_idx[sid]]
            T0C_list.append(sample["T0C"].astype(np.float64))
            joints_list.append(sample["joints_rad"].numpy().astype(np.float64))
            found_ids.append(sid)
        else:
            missing.append(sid)

    if missing:
        print(f"[warn] {len(missing)} scene_ids not found in dataset: {missing[:5]}")

    return T0C_list, joints_list, np.array(found_ids)


def load_robot_kin(robot_name: str = "owi535"):
    """
    Load robot kinematics the same way run_eval_craves.py does:

        import robot_renderer as rr_pkg
        robot_entry = rr_pkg.registry.get_robot_entry(cfg.robot.name)
        # robot_kin is then accessed via adapter.estimator.robot_kin,
        # which is built inside build_estimator using the same registry entry.

    The registry adapter provides all kinematics needed for get_joint_R_t().
    """
    import robot_renderer as rr_pkg
    robot_entry = rr_pkg.registry.get_robot_entry(robot_name)

    adapter_cls = robot_entry["adapter_cls"]
    kin = adapter_cls(robot_entry["urdf"])
    print(f"[info] Loaded robot_kin for '{robot_name}' from {robot_entry['urdf']}")
    return kin


# Main evaluation

def evaluate(npz_path: str, data_folder: str, robot_name: str = "owi535"):
    # load predictions
    print(f"Loading predictions from {npz_path} ...")
    npz = np.load(npz_path, allow_pickle=False)
    scene_ids = npz["scene_ids"].astype(str)   # (428,)
    pred_poses = npz["poses"].astype(np.float64)  # (428, 4, 4)
    pred_kp3d  = npz["kp3d"].astype(np.float64)   # (428, 17, 3) in metres
    has_kp3d   = bool(npz["has_kp3d"][0])

    if not has_kp3d:
        raise RuntimeError(
            "TCO_keypoints_3d was not available when step1 ran (all zeros). "
            "Cannot compute proper ADD without predicted keypoint positions."
        )

    pred_joint_kp3d = pred_kp3d[:, :3, :]   # (428, 3, 3) - Base, Elbow, Wrist

    # load GT
    print(f"Loading CRAVES GT from {data_folder} ...")
    T0C_gt_list, joints_gt_list, matched_ids = load_craves_gt(data_folder, scene_ids)
    n = len(matched_ids)
    print(f"Matched {n} / {len(scene_ids)} frames")

    # load robot_kin for GT FK
    robot_kin = load_robot_kin(robot_name)

    # Build scene_id -> prediction index map
    sid_to_pred_idx = {sid: i for i, sid in enumerate(scene_ids)}

    per_frame_add_mm = []
    records = []

    for k, sid in enumerate(matched_ids):
        pi = sid_to_pred_idx[sid]

        R_gt = T0C_gt_list[k][:3, :3]
        t_gt = T0C_gt_list[k][:3, 3]

        # Predicted joint positions in camera space (from RoboPose TCO_keypoints_3d)
        pts_cam_pred = pred_joint_kp3d[pi]           # (3, 3) metres

        # GT joint positions: FK with GT joints, transformed by GT pose
        R_list, t_list = robot_kin.get_joint_R_t(joints_gt_list[k])
        fk_kp = np.asarray(t_list, dtype=np.float64)[FK_JOINT_KP_INDICES]  # (3, 3)
        pts_cam_gt = (R_gt @ fk_kp.T).T + t_gt      # (3, 3) metres

        # Per-keypoint ADD in mm, then mean over 3 joints
        add_per_kp = np.linalg.norm(pts_cam_pred - pts_cam_gt, axis=1) * 1000.0
        mean_add   = float(add_per_kp.mean())
        per_frame_add_mm.append(mean_add)

        # Also store base-only translation error for cross-check against summary.txt
        t_pred = pred_poses[pi][:3, 3]
        trans_norm_mm = float(np.linalg.norm(t_pred - t_gt) * 1000.0)

        records.append({
            "scene_id":      sid,
            "add_mean_mm":   mean_add,
            "add_base_mm":   float(add_per_kp[0]),
            "add_elbow_mm":  float(add_per_kp[1]),
            "add_wrist_mm":  float(add_per_kp[2]),
            "trans_norm_mm": trans_norm_mm,   # should avg to ~13.1 mm
        })

    per_frame_add_mm = np.array(per_frame_add_mm)
    metrics = compute_add_metrics(per_frame_add_mm)

    # Published RoboPose base-translation norm is approximately 13.1 mm.
    trans_norms = np.array([r["trans_norm_mm"] for r in records])
    print(f"\n[sanity] mean trans_norm_mm = {trans_norms.mean():.2f}  (expect ~13.1)")

    return metrics, pd.DataFrame(records), n


def print_results(metrics: dict, n_frames: int):
    print()
    print("=" * 62)
    print("RoboPose CRAVES-lab  -  Joint-ADD Metrics (iteration=10)")
    print("=" * 62)
    print(f"  Frames evaluated       : {n_frames}")
    print(f"  Mean joint-ADD (mm)    : {metrics['add_mean_mm']:.2f}"
          f"  [median={metrics['add_median_mm']:.2f}  P90={metrics['add_p90_mm']:.2f}]")
    print(f"  ADD AUC @100mm         : {metrics['add_auc_100mm']:.4f}")
    print(f"  ADD AUC @400mm         : {metrics['add_auc_400mm']:.4f}")
    print(f"  ADD @100mm             : {metrics['add_at_100mm']:.4f}")
    print(f"  ADD @50mm              : {metrics['add_at_50mm']:.4f}")
    print()
    print("  Reference (for comparison):")
    print("    RoboPose  base trans norm (no rot)  : 13.1 mm")
    print("    CtRNet    DREAM/Panda (7 joints)    :  9.6 mm  AUC=0.908")
    print("    CtRNet    Baxter (EE only)          : 63.8 mm  AUC=0.839")
    print("=" * 62)


# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz",        required=True,
                   help="Path to robopose_pred_iter10.npz")
    p.add_argument("--data",       required=True,
                   help="Path to CRAVES test_20181024/ folder")
    p.add_argument("--robot_name", default="owi535",
                   help="Robot name as registered in robot_renderer registry "
                        "(default: owi535)")
    p.add_argument("--out_csv",    default="robopose_craves_add_per_frame.csv")
    p.add_argument("--out_json",   default="robopose_craves_add_summary.json")
    args = p.parse_args()

    metrics, df, n = evaluate(args.npz, args.data, args.robot_name)
    print_results(metrics, n)

    df.to_csv(args.out_csv, index=False)
    print(f"\nPer-frame results  -> {args.out_csv}")

    with open(args.out_json, "w") as f:
        json.dump({"n_frames": n, **metrics}, f, indent=2)
    print(f"Summary metrics    -> {args.out_json}")
