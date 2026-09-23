#!/usr/bin/env python3
"""Rotation-error signature check for the MaskVal gate's near-symmetry blind spot.

If the silhouette-IoU gate has a near-symmetry blind spot, frames that SURVIVE the
gate (high IoU) while carrying large pose error (high ADD) should cluster at ~180 deg
rotation error. This script tests that signature on saved runs and contrasts the
survivors against a control group (high ADD, low IoU) where gross failures of any
kind land.

Inputs:
  --results    per-model run dirs holding results.json / results.json.zip
               (queries carry frame_id, est_pose, gt_pose, add_m_mm, pnp_failed)
  --iou-cache  gating iou_cache/<model>__<dataset>.json with aligned lists
               {iou: [...], add: [...], n_abstain: int}
               (written by analysis/gating/risk_coverage_from_masks.py)

Method per model x dataset:
  1. recompute per-frame geodesic rotation error from est_pose / gt_pose
     (same formula as evaluation/run_eval_craves.py:_geodesic_rot_error_deg)
  2. join with the cached IoU by frame order; alignment is checked, not assumed:
     the cached add list must match the queries' add_m_mm elementwise
  3. blind-spot candidates: IoU >= dataset-median IoU AND ADD > 100 mm
     control group:          IoU <  dataset-median IoU AND ADD > 100 mm
  4. report rotation-error histograms and the share within [160, 180] deg
     for both groups; enrichment of ~180 flips among survivors vs control
     is the blind-spot signature.

Datasets whose queries lack gt_pose (Baxter: end-effector-only metric, no
rotational component) are skipped with a note.

    python analysis/rotation_blindspot_check.py --results <results_root> \
        --iou-cache <figures/gating/iou_cache> --out rotation_blindspot.json
"""

import argparse
import glob
import json
import math
import os
import zipfile

import numpy as np

ADD_FAIL_MM = 100.0          # failure tolerance used for the gate's AUROC
FLIP_BAND_DEG = (160.0, 180.0)


def geodesic_rot_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Same formula as evaluation/run_eval_craves.py:_geodesic_rot_error_deg."""
    cos = (np.trace(R_gt.T @ R_pred) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def load_results(path: str) -> dict:
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("results.json"))
    with open(path) as f:
        return json.load(f)


def find_run(results_root: str, model: str, dataset: str):
    """Locate the run dir under results/<model>/ whose summary.dataset matches."""
    model_dir = os.path.join(results_root, model)
    for cand in sorted(glob.glob(os.path.join(model_dir, "*"))):
        for name in ("results.json", "results.json.zip"):
            p = os.path.join(cand, name)
            if os.path.isfile(p):
                try:
                    d = load_results(p)
                except Exception:
                    continue
                if d.get("summary", {}).get("dataset") == dataset:
                    return p, d
    return None, None


def histogram(values, bin_width=10.0):
    bins = np.arange(0.0, 180.0 + bin_width, bin_width)
    counts, edges = np.histogram(values, bins=bins)
    return {f"{int(edges[i])}-{int(edges[i+1])}": int(counts[i]) for i in range(len(counts))}


def analyse(model: str, dataset: str, cache: dict, results: dict) -> dict:
    queries = results["queries"]
    iou = np.asarray(cache["iou"], dtype=float)
    add_cache = np.asarray(cache["add"], dtype=float)
    add = np.asarray([q.get("add_m_mm", np.nan) for q in queries], dtype=float)

    if len(iou) != len(queries):
        return {"skipped": f"cache/queries length mismatch ({len(iou)} vs {len(queries)})"}
    # Check frame alignment: cached ADD must reproduce the queries' ADD elementwise.
    finite = np.isfinite(add) & np.isfinite(add_cache)
    if not np.allclose(add[finite], add_cache[finite], atol=1e-3):
        return {"skipped": "cached ADD does not match queries elementwise (alignment unproven)"}

    rot = np.full(len(queries), np.nan)
    n_no_gt = 0
    for i, q in enumerate(queries):
        est, gt = q.get("est_pose"), q.get("gt_pose")
        if q.get("pnp_failed") or est is None or gt is None:
            n_no_gt += gt is None
            continue
        rot[i] = geodesic_rot_error_deg(np.asarray(est)[:3, :3], np.asarray(gt)[:3, :3])
    if n_no_gt == len(queries):
        return {"skipped": "queries carry no gt_pose (end-effector-only metric?)"}

    valid = np.isfinite(rot) & np.isfinite(iou) & np.isfinite(add)
    iou_med = float(np.median(iou[valid]))
    fail = valid & (add > ADD_FAIL_MM)
    survivors = fail & (iou >= iou_med)   # passed the gate, pose still wrong
    control = fail & (iou < iou_med)      # caught by the gate

    def group_stats(mask):
        r = rot[mask]
        if len(r) == 0:
            return {"n": 0}
        in_band = np.mean((r >= FLIP_BAND_DEG[0]) & (r <= FLIP_BAND_DEG[1]))
        return {
            "n": int(len(r)),
            "rot_err_median_deg": float(np.median(r)),
            "share_160_180_deg": float(in_band),
            "histogram_10deg": histogram(r),
        }

    return {
        "n_frames": int(valid.sum()),
        "iou_median": iou_med,
        "add_fail_threshold_mm": ADD_FAIL_MM,
        "survivors_high_iou_high_add": group_stats(survivors),
        "control_low_iou_high_add": group_stats(control),
        "signature_confirmed": bool(
            survivors.sum() >= 10
            and group_stats(survivors).get("share_160_180_deg", 0.0)
            > 2.0 * group_stats(control).get("share_160_180_deg", 0.0)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--iou-cache", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for cache_path in sorted(glob.glob(os.path.join(args.iou_cache, "*.json"))):
        stem = os.path.splitext(os.path.basename(cache_path))[0]
        model, dataset = stem.split("__", 1)
        with open(cache_path) as f:
            cache = json.load(f)
        path, results = find_run(args.results, model, dataset)
        if results is None:
            report[stem] = {"skipped": "no matching run dir found"}
            continue
        print(f"[check] {stem}  ({path})")
        report[stem] = analyse(model, dataset, cache, results)
        s = report[stem].get("survivors_high_iou_high_add", {})
        c = report[stem].get("control_low_iou_high_add", {})
        if s.get("n"):
            print(f"        survivors n={s['n']:>6}  160-180deg share={s['share_160_180_deg']:.3f}"
                  f"   control n={c.get('n', 0):>6}  share={c.get('share_160_180_deg', float('nan')):.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
