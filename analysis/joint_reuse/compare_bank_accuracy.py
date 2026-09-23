"""Accuracy comparison: stock nemo vs nemo_bank, paired per frame.

The speed economics live in predict_bank.py; this script answers the three accuracy
falsifiers for the perturbation-sweep -> reuse-bank chain:

1. Aggregates: summary add_* deltas stock -> bank. Run-to-run noise floor (tripwire,
   bank inert): ~0.0007 AUC@100, ~0.2 mm mean ADD. Deltas beyond that are real.
2. Mechanism: per-frame delta-ADD on HIT frames against realized d_surf - the perturbation sweep's
   excess ~ 0 predicts reuse costs only the geometric displacement. MISS frames of
   the same run are the noise control. Failure-rate shift on hit frames is the sweep's own
   gate quantity (<= 5pp).
3. Clusters: failure rate grouped by the bank entry that served the frame - one
   unlucky entry must not degrade its neighbourhood.

Pure stdlib; runs anywhere. Pairing is by frame_id; runs are matched by
summary.dataset and classified by the presence of reuse_bank diagnostics.

    python analysis/joint_reuse/compare_bank_accuracy.py outputs/nemo_or_tripwire_dir outputs/nemo_bank_dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

NOISE_AUC100 = 0.0007   # tripwire panda fresh vs bank0 (bank provably inert)
NOISE_MEAN_MM = 0.2


def load_queries(path: str):
    rj = Path(path)
    if rj.is_dir():
        rj = rj / "results.json"
    with open(rj) as f:
        data = json.load(f)
    return data["summary"].get("dataset", rj.parent.name), data["summary"], data["queries"], rj


def _bank_block(q) -> dict:
    return (q.get("diagnostics") or {}).get("reuse_bank") or {}


def _add_m(q) -> float:
    """ADD in metres: panda-style runs carry add_m, baxter-style only add_m_mm."""
    if "add_m" in q:
        return q["add_m"]
    return q.get("add_m_mm", 0.0) / 1000.0


def _failed(q, fail_m: float) -> bool:
    return bool(q.get("pnp_failed")) or _add_m(q) > fail_m


def _fail_rate(qs, fail_m: float) -> float:
    return sum(1 for q in qs if _failed(q, fail_m)) / max(len(qs), 1)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run dirs: one stock + one nemo_bank per dataset")
    ap.add_argument("--tolerance", type=float, default=24.3)
    ap.add_argument("--fail-mm", type=float, default=100.0,
                    help="failure threshold, matches the ADD@100 metric")
    args = ap.parse_args()
    fail_m = args.fail_mm / 1000.0

    stock, bank = {}, {}
    for r in args.runs:
        ds, summary, qs, rj = load_queries(r)
        dst = bank if any(_bank_block(q) for q in qs) else stock
        if ds in dst:
            sys.exit(f"duplicate {'bank' if dst is bank else 'stock'} run for {ds}: "
                     f"{dst[ds][2].parent.name} and {rj.parent.name}")
        dst[ds] = (summary, qs, rj)

    for ds in sorted(stock):
        if ds not in bank:
            print(f"\n== {ds}: no bank run given - skipping")
            continue
        s_sum, s_qs, _ = stock[ds]
        _, b_qs, _ = bank[ds]
        s_by_id = {q["frame_id"]: q for q in s_qs}
        b_by_id = {q["frame_id"]: q for q in b_qs}
        common = sorted(set(s_by_id) & set(b_by_id))
        if len(common) != len(b_qs) or len(common) != len(s_qs):
            print(f"\n== {ds}: WARNING frame sets differ "
                  f"(stock {len(s_qs)}, bank {len(b_qs)}, paired {len(common)})")
        else:
            print(f"\n== {ds} ({len(common)} paired frames) ==")

        # 1. aggregates
        print("  aggregates (stock -> bank, delta):")
        for k in sorted(k for k in s_sum if k.startswith("add_")):
            if k in bank[ds][0] and isinstance(s_sum[k], (int, float)):
                d = bank[ds][0][k] - s_sum[k]
                flag = ""
                if k == "add_auc_100mm" and abs(d) > NOISE_AUC100:
                    flag = "  <- beyond tripwire noise"
                if k == "add_mean_mm" and abs(d) > NOISE_MEAN_MM:
                    flag = "  <- beyond tripwire noise"
                print(f"    {k:16s} {s_sum[k]:.4f} -> {bank[ds][0][k]:.4f}  ({d:+.4f}){flag}")

        # 2. mechanism: delta-ADD on hits vs realized d_surf; misses = control
        pairs = [(s_by_id[f], b_by_id[f]) for f in common]
        hits = [(s, b) for s, b in pairs if _bank_block(b).get("hit")]
        misses = [(s, b) for s, b in pairs if not _bank_block(b).get("hit")]
        dadd = lambda s, b: (_add_m(b) - _add_m(s)) * 1000.0
        if hits:
            d_hit = [dadd(s, b) for s, b in hits]
            dsurf = [_bank_block(b)["d_surf_mm"] for _, b in hits]
            print(f"  hit frames ({len(hits)}): delta-ADD mean {mean(d_hit):+.2f} mm, "
                  f"median {median(d_hit):+.2f} mm | realized d_surf mean {mean(dsurf):.1f} mm | "
                  f"pearson(d_surf, delta-ADD) {_pearson(dsurf, d_hit):.2f}")
            s_fail = _fail_rate([s for s, _ in hits], fail_m)
            b_fail = _fail_rate([b for _, b in hits], fail_m)
            gate = "within A1 gate (<=5pp)" if (b_fail - s_fail) * 100 <= 5 else "GATE EXCEEDED"
            print(f"    failure rate on hit frames: stock {s_fail:.1%} -> bank {b_fail:.1%} "
                  f"({(b_fail - s_fail) * 100:+.1f}pp)  [{gate}]")
            edges = [0.0, args.tolerance / 3, 2 * args.tolerance / 3, args.tolerance]
            for lo, hi in zip(edges, edges[1:]):
                sel = [(s, b) for s, b in hits
                       if lo <= _bank_block(b)["d_surf_mm"] < hi or (hi == edges[-1] and _bank_block(b)["d_surf_mm"] == hi)]
                if sel:
                    print(f"    d_surf [{lo:4.1f},{hi:4.1f}]: n={len(sel):5d}  "
                          f"delta-ADD mean {mean(dadd(s, b) for s, b in sel):+.2f} mm  "
                          f"fail {_fail_rate([b for _, b in sel], fail_m):.1%}")
        if misses:
            d_miss = [dadd(s, b) for s, b in misses]
            print(f"  miss frames ({len(misses)}, noise control): "
                  f"delta-ADD mean {mean(d_miss):+.2f} mm, median {median(d_miss):+.2f} mm")

        # 3. clusters per bank entry: bank failure vs STOCK failure on the same
        # frames - an intrinsically hard region must not count against reuse.
        by_entry: dict[int, list] = {}
        for s, b in pairs:
            idx = _bank_block(b).get("entry_idx")
            if idx is not None:
                by_entry.setdefault(idx, []).append((s, b))
        overall = _fail_rate([b for _, b in pairs], fail_m)
        if not by_entry:
            print("  entries: no entry_idx in diagnostics (pre-instrumentation run) - "
                  "cluster check skipped")
            continue
        big = {i: pb for i, pb in by_entry.items() if len(pb) >= 5}

        def rates(pb):
            return (_fail_rate([b for _, b in pb], fail_m),
                    _fail_rate([s for s, _ in pb], fail_m))

        worst = sorted(big.items(), key=lambda kv: -(rates(kv[1])[0] - rates(kv[1])[1]))[:5]
        hot = [i for i, pb in big.items()
               if (lambda br, sr: br - sr > 0.10 and br > 2 * sr)(*rates(pb))]
        print(f"  entries: {len(by_entry)} total, {len(big)} serving >=5 frames | "
              f"run failure rate {overall:.1%}")
        if worst:
            print("    worst entries by excess failure (bank vs stock, same frames):")
            for i, pb in worst:
                br, sr = rates(pb)
                print(f"      entry {i:4d}: bank {br:.1%} vs stock {sr:.1%}  (n={len(pb)})")
        print(f"    entries >10pp excess and >2x stock: {len(hot)}"
              + ("  <- CHECK: reuse may be poisoning a region" if hot else ""))


if __name__ == "__main__":
    main()
