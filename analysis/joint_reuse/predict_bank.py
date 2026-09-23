"""Predicted vs realized reuse-bank economics, per dataset.

Prediction is stock-only: replay the online-bank policy (covering_analysis.py)
over the STOCK run's joint trajectory and price it with stock per-stage timings
(T_miss = median frame total, T_hit = median total-minus-render-minus-encode). The
geometry is the same construction as the estimator's ReuseBank, so the predicted bank
size must equal the realized one exactly - a mismatch means the two geometry
implementations or the frame orders diverged.

Realized numbers come from the matching nemo_bank run, matched by summary.dataset.
The realized speedup is priced against the bank run's OWN miss frames - a miss is the
full stock pipeline plus a microseconds lookup, measured on the same node in the same
job, so N x median(miss totals) / sum(all totals) is a same-node ratio. The external
stock run's total is printed for reference but is not comparable across nodes/jobs
(the timing README measured 20-25% drift node-to-node on identical rows).

    python analysis/joint_reuse/predict_bank.py outputs/nemo_* --tolerance 24.3

Pass one stock and (optionally) one bank run dir per dataset, in any mix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze_perturbation import RobotGeometry  # noqa: E402
from covering_analysis import build_clouds, online_bank  # noqa: E402


def load_run(path: str):
    """(data, config, results_path) from a run dir or a results.json path."""
    rj = Path(path)
    if rj.is_dir():
        rj = rj / "results.json"
    cfg_path = rj.with_name(rj.name.replace(".json", ".config.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    with open(rj) as f:
        data = json.load(f)
    return data, cfg, rj


def _pricing(queries) -> tuple[float, float]:
    """(T_miss, T_hit) in ms: median frame total, and median per-frame residual
    (total minus render minus encode) - i.e. what a bank hit would have cost."""
    tot, resid = [], []
    for q in queries:
        t = (q.get("diagnostics") or {}).get("timings_ms") or {}
        if "total" in t:
            tot.append(t["total"])
            resid.append(t["total"] - t.get("render", 0.0) - t.get("encode", 0.0))
    if not tot:
        return float("nan"), float("nan")
    return float(np.median(tot)), float(np.median(resid))


def _fmt_time(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run dirs (stock nemo and/or nemo_bank)")
    ap.add_argument("--tolerance", type=float, default=24.3,
                    help="d_surf mm bank tolerance (= A1 delta*)")
    args = ap.parse_args()

    stock, bank = {}, {}
    for r in args.runs:
        data, cfg, rj = load_run(r)
        ds = data["summary"].get("dataset", rj.parent.name)
        dst = bank if (cfg.get("estimator") or {}).get("type") == "nemo_bank" else stock
        if ds in dst:
            print(f"[warn] duplicate {'bank' if dst is bank else 'stock'} run for {ds}: "
                  f"keeping {dst[ds][2].parent.name}, ignoring {rj.parent.name}")
            continue
        dst[ds] = (data, cfg, rj)

    for ds in sorted(stock):
        data, cfg, rj = stock[ds]
        qs = data["queries"]
        N = len(qs)
        configs = np.array([q["joints_rad"] for q in qs], dtype=np.float64)
        robot = (cfg.get("robot") or {}).get("name")
        geom = RobotGeometry(robot)
        clouds, W = build_clouds(geom, configs)
        entries, _ = online_bank(clouds, W, args.tolerance)
        k_pred = len(entries)

        t_miss, t_hit = _pricing(qs)
        stock_total = sum((q.get("diagnostics") or {}).get("timings_ms", {}).get("total", 0.0)
                          for q in qs) / 1000.0
        pred_total = (k_pred * t_miss + (N - k_pred) * t_hit) / 1000.0

        print(f"\n== {ds} (robot {robot}, N={N}, tau={args.tolerance:g} mm) ==")
        print(f"  predicted: bank {k_pred} ({N / max(k_pred, 1):.0f}x fewer builds), "
              f"total {_fmt_time(pred_total)} vs stock {_fmt_time(stock_total)} "
              f"-> {stock_total / max(pred_total, 1e-9):.1f}x")
        print(f"  pricing:   T_miss={t_miss:.0f} ms  T_hit={t_hit:.0f} ms "
              f"(stock medians)")

        if ds not in bank:
            continue
        bdata, _, brj = bank[ds]
        bqs = bdata["queries"]
        blocks = [(q.get("diagnostics") or {}).get("reuse_bank") or {} for q in bqs]
        hits = sum(1 for b in blocks if b.get("hit"))
        k_real = blocks[-1].get("bank_size", 0) if blocks else 0
        totals = [(q.get("diagnostics") or {}).get("timings_ms", {}).get("total", 0.0)
                  for q in bqs]
        real_total = sum(totals) / 1000.0
        miss_tot = [t for t, b in zip(totals, blocks) if not b.get("hit") and t > 0]
        t_miss_own = float(np.median(miss_tot)) if miss_tot else float("nan")
        same_node = len(bqs) * t_miss_own / 1000.0 / max(real_total, 1e-9)
        check = "MATCH" if k_real == k_pred else "MISMATCH - metric or frame order diverged"
        warn = f"  WARNING: N differs (stock {N} vs bank {len(bqs)})" if len(bqs) != N else ""
        print(f"  realized:  bank {k_real}, {hits}/{len(bqs)} hits, total {_fmt_time(real_total)} "
              f"-> {same_node:.1f}x same-node (N x own miss median {t_miss_own:.0f} ms)   [K {check}]{warn}")
        print(f"             ({stock_total / max(real_total, 1e-9):.1f}x vs external stock dir - "
              f"cross-run, not comparable)")


if __name__ == "__main__":
    main()
