"""Backfill DREAM-style keypoint blocks into hydra/craves results.json.

Adds to every query:
    objects           [{"class", "keypoints": [{name, location, projected_location}]}]
    objects_detected  same but name + projected_location only; null on PnP failure

location = gt_pose @ FK(joints_rad); both projections use the query's K_eval.
Keypoints are each dataset's ADD set: all FK link origins for hydra
(run_eval_hydra.py), [Base, Elbow, Wrist] for craves (run_eval_craves.py).
panda_orb and baxter files are skipped.

    python analysis/add_keypoints.py <results_dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CRAVES_FK_JOINT_KP_INDICES = [2, 3, 4]                 # run_eval_craves.py

_robots: dict = {}


def _import_registry():
    """Import robot_renderer.registry, falling back to the source tree."""
    try:
        from robot_renderer import registry  # installed in the ROBOP env
    except ImportError as exc:
        rr_src = Path(__file__).resolve().parents[1] / "external" / "robot-renderer" / "src"
        if not rr_src.exists():
            raise ImportError(
                f"robot_renderer not importable and {rr_src} not found. "
                f"Run `git submodule update --init external/robot-renderer`, "
                f"or install robot-renderer."
            ) from exc
        sys.path.insert(0, str(rr_src))
        from robot_renderer import registry
    registry._register_builtins()
    return registry


def _adapter(robot_name: str):
    if robot_name not in _robots:
        entry = _import_registry().get_robot_entry(robot_name)
        _robots[robot_name] = entry["adapter_cls"](entry["urdf"])
    return _robots[robot_name]


def _fk_keypoints(dataset: str, joints_rad):
    """(robot, names, base-frame points) for the dataset's ADD keypoints."""
    if dataset.startswith("hydra_"):
        robot, kp_idx = dataset[len("hydra_"):], None
    elif dataset == "craves":
        robot, kp_idx = "owi535", CRAVES_FK_JOINT_KP_INDICES
    else:
        return None
    ad = _adapter(robot)
    names = [ad._robot.links[i].name for i in ad._link_indices]
    _, t = ad.get_joint_R_t(np.asarray(joints_rad, dtype=np.float64))
    if kp_idx is not None:
        names = [names[i] for i in kp_idx]
        t = t[kp_idx]
    return robot, names, np.asarray(t, dtype=np.float64)


def _apply(pose, pts):
    """Transform base-frame points (N,3) by a 4x4 cam-from-base pose."""
    pose = np.asarray(pose, dtype=np.float64)
    return (pose[:3, :3] @ pts.T).T + pose[:3, 3]


def _project(K, pts):
    """Pinhole projection of camera-frame points (N,3) to pixels (N,2)."""
    uv = (np.asarray(K, dtype=np.float64) @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def process_file(path: Path, dry_run: bool) -> bool:
    data = json.loads(path.read_text())
    dataset = data.get("summary", {}).get("dataset", "")
    if not (dataset.startswith("hydra_") or dataset == "craves"):
        return False

    for q in data["queries"]:
        robot, names, base = _fk_keypoints(dataset, q["joints_rad"])
        cam_gt = _apply(q["gt_pose"], base)
        uv_gt = _project(q["K_eval"], cam_gt)
        q["objects"] = [{"class": robot, "keypoints": [
            {"name": n, "location": p.tolist(), "projected_location": uv.tolist()}
            for n, p, uv in zip(names, cam_gt, uv_gt)]}]
        if q.get("est_pose") is None:
            q["objects_detected"] = None
        else:
            uv_pred = _project(q["K_eval"], _apply(q["est_pose"], base))
            q["objects_detected"] = [{"class": robot, "keypoints": [
                {"name": n, "projected_location": uv.tolist()}
                for n, uv in zip(names, uv_pred)]}]

    if dry_run:
        q0 = data["queries"][0]
        kp0 = q0["objects"][0]["keypoints"][0]
        print(f"  [dry-run] {dataset}: {len(data['queries'])} queries, "
              f"{len(q0['objects'][0]['keypoints'])} keypoints/frame")
        print(f"    frame0 kp[0] {kp0['name']}: location={np.round(kp0['location'], 3).tolist()}  "
              f"projected={np.round(kp0['projected_location'], 1).tolist()}")
    else:
        path.write_text(json.dumps(data, separators=(",", ":")))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", type=Path, help="dir to scan for results.json")
    ap.add_argument("--dry-run", action="store_true", help="preview frame 0, do not write")
    args = ap.parse_args()

    files = sorted(args.results_dir.rglob("results.json"))
    n = 0
    for f in files:
        if process_file(f, args.dry_run):
            print(f"{'checked' if args.dry_run else 'updated'}: {f}")
            n += 1
    print(f"\n{n} results.json {'checked' if args.dry_run else 'updated'} ({len(files)} scanned).")


if __name__ == "__main__":
    main()
