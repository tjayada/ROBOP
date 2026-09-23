#!/usr/bin/env python3
"""Render ground-truth robot silhouettes from an evaluation JSON's GT poses.

For every query, the mesh is FK-posed and rendered at the stored ``gt_pose``
(PyTorch3D, exactly as analysis/visualize_results.py), and the silhouette is
written as a column-major RLE. Output mirrors robot-detector's detection schema
(found/bbox_xyxy/segmentation), so existing mask loaders read it unchanged.

``gt_pose`` is ground truth (model-independent), so any full-coverage results
JSON for a dataset works - panda-orb, Hydra and CRAVES all store the same keys.
Baxter is unsupported: it has no GT base pose (only an end-effector point).

Long runs journal each finished frame to ``<output>.partial.jsonl`` and resume
with --resume (checkpoint pattern mirrors robot-detector run_detect.py).
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from eval_utils import select_eval_frame_indices  # noqa: E402

from robot_renderer.config import ViewConfig  # noqa: E402
from robot_renderer.robot_renderer import RobotRenderer  # noqa: E402

# OpenCV/BOP -> PyTorch3D axis flip (mirror of visualize_results._pose_to_R_T).
_C3 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def _pose_to_R_T(T_m2c: np.ndarray):
    R = (T_m2c[:3, :3].T @ _C3).astype(np.float32)
    T = (_C3 @ T_m2c[:3, 3]).astype(np.float32)
    return R, T


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Column-major RLE, exact inverse of eval_utils._rle_to_mask.

    Runs are read starting at value 0, so a leading set pixel emits a
    zero-length first count. ``size`` is [H, W]; ``counts`` is Fortran order.
    """
    H, W = mask.shape
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0] == 1:
        counts = [0] + counts
    return {"counts": counts, "size": [H, W]}


def _get_K(q: dict) -> Optional[np.ndarray]:
    for key in ("K", "K_eval"):
        if q.get(key) is not None:
            return np.asarray(q[key], dtype=np.float32)
    return None


# Checkpoint journal (mirrors robot-detector run_detect.py)

def _open_journal(path: Path, signature: dict, resume: bool):
    """Open a new checkpoint journal, or validate and resume an existing one."""
    if not resume:
        if path.exists():
            raise FileExistsError(
                f"Checkpoint already exists: {path}. Use --resume to continue it, "
                "or remove it to start over."
            )
        journal = open(path, "x")
        journal.write(json.dumps({"run": signature}, separators=(",", ":")) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
        return journal, []

    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found to resume: {path}")
    entries = []
    with open(path, "rb+") as raw:
        header_line = raw.readline()
        if not header_line:
            raise ValueError(f"Checkpoint is empty: {path}")
        if json.loads(header_line).get("run") != signature:
            raise ValueError(
                f"Checkpoint settings do not match this run: {path}. "
                "Use the original results file and selection, or start over."
            )
        file_size = os.fstat(raw.fileno()).st_size
        while line := raw.readline():
            line_start = raw.tell() - len(line)
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if raw.tell() != file_size:
                    raise ValueError(f"Invalid checkpoint record in {path}") from None
                raw.truncate(line_start)   # drop a torn final record
                break
    ids = [e["image_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Checkpoint contains duplicate image IDs: {path}")
    return open(path, "a"), entries


def _append_journal(journal, entry: dict, sync: bool) -> None:
    journal.write(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n")
    journal.flush()
    if sync:
        os.fsync(journal.fileno())


def _write_json_atomic(path: Path, data: dict) -> None:
    """Replace path only after a complete document is durable; ``.gz`` gzips it."""
    gz = path.suffix == ".gz"
    text = json.dumps(data, separators=(",", ":"), allow_nan=False)
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(gzip.compress(text.encode()) if gz else text.encode())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Results JSON from any run_eval_* script (its GT poses are used)")
    parser.add_argument("--output", required=True,
                        help="Output path; .json.gz is written gzipped and compact")
    parser.add_argument("--robot", default=None,
                        help="robot-renderer registry name; default: robot.name from the "
                             "sibling <results>.config.yaml")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Render N evenly-spaced frames; >= total (or omitted) renders all")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output>.partial.jsonl (same --results and --num-samples)")
    parser.add_argument("--checkpoint-every", type=int, default=10, metavar="N",
                        help="Durably fsync the journal every N finished frames")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        results = json.load(f)
    all_queries = results["queries"]

    robot_name = args.robot
    config_path = results_path.with_suffix(".config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            run_cfg = yaml.safe_load(f)
        robot_name = robot_name or run_cfg.get("robot", {}).get("name")
    if robot_name is None:
        sys.exit(f"ERROR: --robot not given and no sibling config at {config_path}")

    indices = select_eval_frame_indices(len(all_queries), args.num_samples)
    queries = [all_queries[i] for i in indices]
    if not queries:
        sys.exit("ERROR: no frames in the results JSON")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(f"{out_path}.partial.jsonl")
    signature = {
        "version": 1,
        "results": results_path.name,
        "robot": robot_name,
        "total_queries": len(all_queries),
        "num_samples": args.num_samples,
    }
    journal, entries = _open_journal(journal_path, signature, resume=args.resume)
    done_ids = {e["image_id"] for e in entries}
    remaining = set(range(len(queries))) - done_ids
    if entries:
        print(f"Resuming: {len(done_ids)} done, {len(remaining)} remaining")
    print(f"Rendering GT masks for {len(remaining)} frames ({robot_name})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Renderer built once; PT3D cameras rebuilt whenever K or (H, W) changes
    # (CRAVES has per-frame K; Hydra pools measurements with different K).
    pt3d = None
    pt3d_built_for = None

    try:
        for image_id, q in enumerate(queries):
            if image_id not in remaining:
                continue
            frame_id = str(q.get("frame_id") or Path(q["image_path"]).stem)

            gt_pose = q.get("gt_pose")
            entry = {"image_id": image_id, "frame_id": frame_id,
                     "found": False, "bbox_xyxy": None, "segmentation": None}

            if gt_pose is not None:
                gt_pose = np.asarray(gt_pose, dtype=np.float32)
                K = _get_K(q)
                H, W = int(q["image_size_hw"][0]), int(q["image_size_hw"][1])
                if K is None:
                    sys.exit(f"ERROR: {frame_id}: no intrinsics in the results JSON")

                if pt3d is None:
                    rr = RobotRenderer(name=robot_name, K=K,
                                       config=ViewConfig(render_size=max(H, W)), device=device)
                    rr.to(device)
                    pt3d = rr._renderer
                frame_key = (K.tobytes(), H, W)
                if frame_key != pt3d_built_for:
                    pt3d._build_pt3d_renderers(K, (H, W))
                    pt3d_built_for = frame_key

                joints = np.asarray(q["joints_rad"], dtype=np.float32)
                robot_meshes = pt3d.transform_mesh2robot_config(joints)
                R_np, T_np = _pose_to_R_T(gt_pose)
                R = torch.from_numpy(R_np).unsqueeze(0).to(device)
                T = torch.from_numpy(T_np).unsqueeze(0).to(device)
                _, fragments = pt3d.phong_render_batched(robot_meshes, R, T)
                mask = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy()

                ys, xs = np.where(mask)
                if xs.size:
                    entry.update(found=True, segmentation=_mask_to_rle(mask), bbox_xyxy=[
                        float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

            entries.append(entry)
            _append_journal(journal, entry, sync=len(entries) % args.checkpoint_every == 0)
            remaining.discard(image_id)
            if len(entries) % 200 == 0:
                print(f"  [{len(entries)}/{len(queries)}]")
    finally:
        journal.flush()
        os.fsync(journal.fileno())
        journal.close()

    entries.sort(key=lambda e: e["image_id"])
    n_rendered = sum(e["found"] for e in entries)
    out = {
        "source": "gt_mesh_projection",
        "results": results_path.name,
        "robot": robot_name,
        "num_frames": len(entries),
        "num_rendered": n_rendered,
        "detections": entries,
    }
    _write_json_atomic(out_path, out)
    journal_path.unlink()
    print(f"Done. {n_rendered}/{len(entries)} GT masks -> {out_path}")


if __name__ == "__main__":
    main()
