#!/usr/bin/env python3
"""Print ADD mean (mm) and AUC (x100) at 100/400 mm for every results.json under a results dir.

    python analysis/collect_results.py [results_root]   (default: ROBOP/results)
"""
import sys
import json
from pathlib import Path

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "results"

rows = []
for f in sorted(RESULTS.rglob("results.json")):
    if f.parent.name.startswith("shard_"):  # keep only merged, drop per-shard parts
        continue
    s = json.load(f.open()).get("summary", {})
    rows.append((
        str(f.parent.relative_to(RESULTS)),
        s.get("add_mean_mm"),
        s.get("add_auc_100mm"),
        s.get("add_auc_400mm"),
    ))

w = max((len(r[0]) for r in rows), default=3)
print(f"{'dir':<{w}}  {'ADD_mean_mm':>11}  {'AUC@100':>7}  {'AUC@400':>7}")
for name, mean, a100, a400 in rows:
    m = f"{mean:11.2f}" if mean is not None else f"{'n/a':>11}"
    x1 = f"{a100*100:7.2f}" if a100 is not None else f"{'n/a':>7}"
    x4 = f"{a400*100:7.2f}" if a400 is not None else f"{'n/a':>7}"
    print(f"{name:<{w}}  {m}  {x1}  {x4}")
