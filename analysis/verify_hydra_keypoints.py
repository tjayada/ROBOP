"""Self-consistency check for DREAM keypoints written by add_hydra_keypoints.py.

Uses only the results.json (no robot assets). For every Hydra query it verifies:
  1. GT 2D:  reprojecting objects.keypoints.location through K_eval reproduces
             the stored projected_location.
  2. det 2D: mapping those GT 3D points into the est_pose frame and reprojecting
             reproduces objects_detected.keypoints.projected_location.
  3. ADD:    mean 3D distance (GT vs est-mapped) equals the stored add_m_mm.
Fails loudly if any tolerance is exceeded.

    python verify_hydra_keypoints.py <results_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

TOL_PX = 0.5      # projection round-trip (JSON keeps full float precision)
TOL_MM = 0.01     # add_m_mm stored rounded to 3 decimals


def _project(K, pts):
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def check_file(path: Path) -> bool:
    data = json.loads(path.read_text())
    if not data.get("summary", {}).get("dataset", "").startswith("hydra_"):
        return False

    worst_gt_px = worst_det_px = worst_add_mm = 0.0
    n_det = 0
    for q in data["queries"]:
        K = np.asarray(q["K_eval"], float)
        gp = np.asarray(q["gt_pose"], float)
        kps = q["objects"][0]["keypoints"]
        loc_gt = np.asarray([k["location"] for k in kps], float)          # 3D wrt cam
        proj_gt = np.asarray([k["projected_location"] for k in kps], float)

        # 1. GT reprojection round-trip
        worst_gt_px = max(worst_gt_px, np.abs(_project(K, loc_gt) - proj_gt).max())

        if q["objects_detected"] is None:
            continue
        n_det += 1
        ep = np.asarray(q["est_pose"], float)
        # recover base-frame link points by inverting the GT camera transform
        t_links = (np.linalg.inv(gp[:3, :3]) @ (loc_gt.T - gp[:3, 3:4])).T
        cam_pred = (ep[:3, :3] @ t_links.T + ep[:3, 3:4]).T

        # 2. detected reprojection matches stored predicted 2D
        proj_det = np.asarray([k["projected_location"] for k in q["objects_detected"][0]["keypoints"]], float)
        worst_det_px = max(worst_det_px, np.abs(_project(K, cam_pred) - proj_det).max())

        # 3. ADD invariant
        add = np.mean(np.linalg.norm(cam_pred - loc_gt, axis=1)) * 1000
        worst_add_mm = max(worst_add_mm, abs(add - q["add_m_mm"]))

    ok = worst_gt_px < TOL_PX and worst_det_px < TOL_PX and worst_add_mm < TOL_MM
    print(f"[{'PASS' if ok else 'FAIL'}] {path}")
    print(f"    frames={len(data['queries'])} (pred {n_det}) | "
          f"GT reproj max={worst_gt_px:.4f}px | det reproj max={worst_det_px:.4f}px | "
          f"ADD mismatch max={worst_add_mm:.5f}mm")
    return ok


def main():
    root = Path(sys.argv[1])
    files = sorted(root.rglob("results.json"))
    checked = [f for f in files if check_file(f)]
    print(f"\n{len(checked)} Hydra files checked.")


if __name__ == "__main__":
    main()
