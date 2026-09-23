"""Print ADD metrics for any results.json, including partial shard outputs.

Rebuilds the per-frame error array from the queries (stored add_m_mm, failures
at the 1000 mm sentinel from run_eval_panda_orb.py) and recomputes the summary
metrics, so it works on files whose summary is missing or covers only part of
the eval run. Numbers on a partial file describe that file's frames only.

    python analysis/results_metrics.py results.json [more.json ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_utils import add_auc

FAIL_MM = 1000.0  # failure sentinel, run_eval_panda_orb.py


def per_frame_errors_mm(results: dict) -> tuple[np.ndarray, int]:
    queries = results["queries"]
    err = np.array([FAIL_MM if q["est_pose"] is None else q["add_m_mm"]
                    for q in queries], dtype=np.float64)
    n_fail = sum(q["est_pose"] is None for q in queries)
    return err, n_fail


def main():
    ap = argparse.ArgumentParser(
        description="Print ADD metrics recomputed from a results.json's queries.")
    ap.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()

    for path in args.results:
        results = json.loads(path.read_text())
        err, n_fail = per_frame_errors_mm(results)
        print(f"{path.name}: {err.size} frames ({n_fail} failures)")
        print(f"  ADD mm   mean={np.mean(err):.2f}  median={np.median(err):.2f}  "
              f"p90={np.percentile(err, 90):.2f}  p95={np.percentile(err, 95):.2f}")
        print(f"  ADD-AUC  @100mm={add_auc(err, 100.0):.4f}  @400mm={add_auc(err, 400.0):.4f}")
        print(f"  ADD@50mm={np.mean(err < 50):.4f}  ADD@100mm={np.mean(err < 100):.4f}")


if __name__ == "__main__":
    main()
