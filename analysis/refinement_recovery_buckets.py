#!/usr/bin/env python3
"""Per-bucket refinement recovery on Panda-Orb.

Refinement is a paired, per-frame operation: every refined run is initialised
from the per-frame coarse pose of its matching coarse run. This script joins
each coarse run to its refined run by frame_id, groups the frames by their
COARSE ADD into buckets, and reports, per bucket, the median coarse and refined
ADD plus the recovery rate (share of frames whose refined ADD ends below the
100 mm ceiling).

Failures (est_pose is None) enter at the 1000 mm sentinel, matching
iou_add_detection_curve.py; a frame is kept only if BOTH its coarse and
refined runs report it.

    python analysis/refinement_recovery_buckets.py --results <results_root>

<results_root>/<estimator>/<run>/results.json[.zip]; the run names are in PAIRS.
"""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

FAIL_MM = 1000.0            # failure sentinel, results_metrics.py
CEIL_MM = 100.0            # recovery ceiling (the ADD-AUC failure tolerance)
# bucket lower edges; the open-topped last bucket runs to +inf
EDGES = [25.0, 100.0, 400.0, np.inf]

# (coarse root/sub, refined root/sub) per estimator, Panda-Orb.
# Same subdir mapping as iou_add_detection_curve.py.
PAIRS = {
    "NeMO":      (("nemo_coarse", "panda_orb_nemo_shards_final"),
                  ("nemo_coarse_megapose_refiner", "panda_orb_refine_nemo_shards_final")),
    "MegaPose":  (("megapose_coarse", "panda_orb_megapose_coarse_shards_final"),
                  ("megapose_coarse_megapose_refiner", "panda_orb_megapose_shards_final")),
    "GigaPose":  (("gigapose_coarse", "panda_orb_gigapose_shards_final"),
                  ("gigapose_coarse_megapose_refiner", "panda_orb_gigapose_megapose_shards_final")),
    "FoundPose": (("foundpose_coarse", "panda_orb_foundpose_shards_final"),
                  ("foundpose_coarse_megepose_refiner", "panda_orb_refine_foundpose_shards_final")),
}


def load_results(res, root, sub):
    d = res / root / sub
    if (d / "results.json").exists():
        return json.load(open(d / "results.json"))
    z = d / "results.json.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".json")][0]
            return json.loads(zf.read(name))
    raise FileNotFoundError(d)


def add_by_frame(results):
    """frame_id -> ADD (mm), failures at the sentinel."""
    out = {}
    for q in results["queries"]:
        fid = q.get("frame_id")
        if fid is None:
            continue
        add = FAIL_MM if q.get("est_pose") is None else q.get("add_m_mm")
        if add is not None:
            out[fid] = float(add)
    return out


def bucket_label(lo, hi):
    return f"[{int(lo)},{'inf' if np.isinf(hi) else int(hi)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="results root")
    args = ap.parse_args()

    for name, (coarse, refined) in PAIRS.items():
        c = add_by_frame(load_results(args.results, *coarse))
        r = add_by_frame(load_results(args.results, *refined))
        fids = sorted(set(c) & set(r))
        ca = np.array([c[f] for f in fids])
        ra = np.array([r[f] for f in fids])
        n = len(fids)
        below25 = int(np.sum(ca < 25.0))
        refined_fail = float(np.mean(ra >= CEIL_MM))   # share the refiner leaves >100mm
        print(f"\n{name}  (paired frames n={n}; coarse<25mm: {below25}; "
              f"refined still >100mm: {refined_fail:.1%})")
        print(f"  {'bucket':>12} {'n':>7} {'coarse med':>11} {'refined med':>12} "
              f"{'recover<100':>12}")
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            sel = (ca >= lo) & (ca < hi)
            k = int(sel.sum())
            if k == 0:
                print(f"  {bucket_label(lo, hi):>12} {k:>7}")
                continue
            cm = float(np.median(ca[sel]))
            rm = float(np.median(ra[sel]))
            rec = float(np.mean(ra[sel] < CEIL_MM))
            print(f"  {bucket_label(lo, hi):>12} {k:>7} {cm:>10.0f}m {rm:>11.0f}m "
                  f"{rec:>11.0%}")


if __name__ == "__main__":
    main()
