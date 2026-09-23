#!/usr/bin/env python3
"""Compare two evaluation JSONs for joint-noise no-op or determinism checks.

Accuracy fields and joint-noise diagnostics must match exactly; volatile timing
and memory diagnostics are ignored. Numeric failures report maximum difference.
"""
from __future__ import annotations

import argparse
import json
import sys


def _leaf_diffs(a, b, path=""):
    """Yield (path, a, b) for every differing leaf between two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield (f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>"))
            else:
                yield from _leaf_diffs(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield (f"{path}(len)", len(a), len(b))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from _leaf_diffs(x, y, f"{path}[{i}]")
    elif a != b:
        yield (path, a, b)


def _max_abs_diff(diffs):
    m = 0.0
    for _, a, b in diffs:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            m = max(m, abs(float(a) - float(b)))
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_a")
    ap.add_argument("json_b")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expect-no-joint-noise", action="store_true",
                      help="assert neither run carries a joint_noise block")
    mode.add_argument("--expect-joint-noise", action="store_true",
                      help="assert both runs carry identical joint_noise blocks")
    ap.add_argument("--expect-magnitude", type=float, default=None,
                    help="with --expect-joint-noise: assert requested_deg equals this")
    ap.add_argument("--max-report", type=int, default=5)
    args = ap.parse_args()

    with open(args.json_a) as f:
        run_a = json.load(f)
    with open(args.json_b) as f:
        run_b = json.load(f)

    failures = []

    qa, qb = run_a.get("queries", []), run_b.get("queries", [])
    if len(qa) != len(qb):
        failures.append(f"frame count differs: {len(qa)} vs {len(qb)}")
        qa, qb = [], []

    clamp_frames = 0
    realized = []
    for i, (fa, fb) in enumerate(zip(qa, qb)):
        core_a = {k: v for k, v in fa.items() if k != "diagnostics"}
        core_b = {k: v for k, v in fb.items() if k != "diagnostics"}
        diffs = list(_leaf_diffs(core_a, core_b, f"queries[{i}]"))
        if diffs:
            failures.append(
                f"frame {i} ({fa.get('image_path', '?')}): {len(diffs)} differing fields, "
                f"max_abs_diff={_max_abs_diff(diffs):.3e}; first: {diffs[0]}"
            )

        jn_a = (fa.get("diagnostics") or {}).get("joint_noise")
        jn_b = (fb.get("diagnostics") or {}).get("joint_noise")
        if args.expect_no_joint_noise:
            if jn_a is not None or jn_b is not None:
                failures.append(f"frame {i}: unexpected joint_noise block (A={jn_a is not None}, B={jn_b is not None})")
        else:
            if jn_a is None or jn_b is None:
                failures.append(f"frame {i}: missing joint_noise block (A={jn_a is not None}, B={jn_b is not None})")
            else:
                if jn_a != jn_b:
                    d = list(_leaf_diffs(jn_a, jn_b, f"queries[{i}].joint_noise"))
                    failures.append(f"frame {i}: joint_noise blocks differ; first: {d[0]}")
                if args.expect_magnitude is not None and jn_a["requested_deg"] != args.expect_magnitude:
                    failures.append(f"frame {i}: requested_deg={jn_a['requested_deg']} != {args.expect_magnitude}")
                if jn_a.get("clamp_count", 0) > 0:
                    clamp_frames += 1
                realized.append(jn_a.get("realized_norm_deg", float("nan")))

    sum_a, sum_b = run_a.get("summary", {}), run_b.get("summary", {})
    for k in sorted(set(sum_a) | set(sum_b)):
        if k.startswith("add_") or k == "num_samples":
            if sum_a.get(k) != sum_b.get(k):
                failures.append(f"summary.{k}: {sum_a.get(k)} vs {sum_b.get(k)}")

    if realized:
        print(f"[info] joint_noise: {len(realized)} frames, realized_norm_deg "
              f"min={min(realized):.4f} max={max(realized):.4f}, "
              f"clamped frames: {clamp_frames}/{len(realized)}")

    if failures:
        print(f"FAIL - {len(failures)} problem(s):")
        for line in failures[: args.max_report]:
            print("  -", line)
        if len(failures) > args.max_report:
            print(f"  ... and {len(failures) - args.max_report} more")
        return 1
    print(f"PASS - {len(qa)} frames identical "
          f"({'no joint_noise blocks' if args.expect_no_joint_noise else 'joint_noise blocks identical'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
