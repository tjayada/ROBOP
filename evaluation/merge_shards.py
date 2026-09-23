"""Merge frame_shard=k/M eval shards into one results.json.

Shards must agree on config, detections and provenance (except cuda_device,
which may differ across shards) and tile the frames disjointly. Metrics are
recomputed over the union - per-shard summaries cannot be averaged, ADD-AUC is
global. Runtime/timing is approximate: each shard warms up on its own.

    python evaluation/merge_shards.py \
        --shards outputs/panda_orb_shards/shard_*/results.json \
        --output outputs/panda_orb_merged/results.json
"""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from eval_utils import aggregate_frame_diagnostics, compute_add_metrics


# Summary keys the merge recomputes over the union or sets itself; every other
# summary key is model identity and must match across shards.
_NON_IDENTITY_PREFIXES = ("add_", "refiner_", "timing_", "gpu_peak_", "pnp_", "runtime_")
_NON_IDENTITY_KEYS = ("num_samples", "num_shards")


def _provenance_identity(result: dict) -> dict:
    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("every shard must contain a provenance object")
    identity = dict(provenance)
    identity.pop("timestamp_utc", None)
    identity.pop("cuda_device", None)   # may differ across shards; recorded, not asserted
    return identity


def _static_summary(result: dict) -> dict:
    summary = result.get("summary")
    if not isinstance(summary, dict) or "dataset" not in summary:
        raise ValueError("every shard must contain summary.dataset")
    return {key: value for key, value in summary.items()
            if key not in _NON_IDENTITY_KEYS
            and not key.startswith(_NON_IDENTITY_PREFIXES)}


def _require_same(label: str, values: list) -> None:
    reference = values[0]
    for shard_idx, value in enumerate(values[1:], start=1):
        if value != reference:
            raise ValueError(f"shard {shard_idx} has different {label}")


def _validate_result_identity(shard_results: list) -> None:
    if not shard_results:
        raise ValueError("no shard results supplied")
    _require_same("detections", [r.get("detections") for r in shard_results])
    _require_same("provenance", [_provenance_identity(r) for r in shard_results])
    _require_same("model identity", [_static_summary(r) for r in shard_results])


def _normalized_config(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    normalized.pop("frame_shard", None)
    normalized.pop("save_results", None)
    return normalized


def merge_shard_results(shard_results: list) -> dict:
    """Combine parsed results.json dicts (in shard k-order) into one results dict.

    Reconstructs per-frame ADD from each query's `add_m` (metres; PnP failures
    stored as the 1.0 -> 1000 mm sentinel), exactly the array the single-job
    path feeds compute_add_metrics.
    """
    _validate_result_identity(shard_results)
    queries = [q for r in shard_results for q in r["queries"]]

    dupes = [f for f, n in Counter(q["frame_id"] for q in queries).items() if n > 1]
    if dupes:
        raise ValueError(
            f"shards overlap: {len(dupes)} frame_id(s) appear in more than one shard, "
            f"e.g. {sorted(dupes)[:5]}"
        )

    err_mm = np.array([q["add_m"] * 1000.0 for q in queries], dtype=np.float64)
    metrics = compute_add_metrics(err_mm)
    metrics.update(aggregate_frame_diagnostics([q.get("diagnostics") for q in queries]))

    summary = _static_summary(shard_results[0])
    summary.update(metrics)
    summary["num_samples"] = len(queries)
    summary["num_shards"] = len(shard_results)

    merged_provenance = dict(shard_results[0].get("provenance") or {})
    devices = sorted(
        {(r.get("provenance") or {}).get("cuda_device") for r in shard_results} - {None}
    )
    merged_provenance.pop("cuda_device", None)
    merged_provenance["cuda_devices"] = devices

    return {
        "summary": summary,
        "detections": shard_results[0].get("detections"),
        "provenance": merged_provenance,
        "queries": queries,
    }


def _load_shard(path: str):
    """Return the parsed results, frame_shard value, and required sibling config."""
    p = Path(path)
    results = json.loads(p.read_text())
    cfg_path = p.with_suffix(".config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing sibling config for {p}: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"invalid shard config: {cfg_path}")
    shard = config.get("frame_shard")
    if shard is None:
        raise ValueError(f"{cfg_path} has no frame_shard=k/M value")
    return results, str(shard), config


def _parse_shard(value: str) -> tuple[int, int]:
    try:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError
        k, m = (int(part) for part in parts)
    except ValueError:
        raise ValueError(f"invalid frame_shard {value!r}; expected k/M") from None
    if m < 1 or not 0 <= k < m:
        raise ValueError(f"invalid frame_shard {value!r}; require M >= 1 and 0 <= k < M")
    return k, m


def main():
    ap = argparse.ArgumentParser(description="Merge sharded eval results into one results.json.")
    ap.add_argument("--shards", nargs="+", required=True, help="Per-shard results.json files.")
    ap.add_argument("--output", required=True, help="Where to write the merged results.json.")
    args = ap.parse_args()

    loaded = [_load_shard(path) for path in args.shards]
    parsed_ids = [_parse_shard(shard) for _, shard, _ in loaded]
    ms = {m for _, m in parsed_ids}
    if len(ms) != 1:
        raise ValueError(f"shards disagree on M: {sorted(ms)}")
    m = ms.pop()
    ks = sorted(k for k, _ in parsed_ids)
    if ks != list(range(m)):
        raise ValueError(f"expected shards k=0..{m - 1} exactly once, got {ks}")

    order = sorted(range(len(loaded)), key=lambda i: parsed_ids[i][0])
    results = [loaded[i][0] for i in order]
    configs = [loaded[i][2] for i in order]
    _require_same("resolved config", [_normalized_config(config) for config in configs])

    merged = merge_shard_results(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, separators=(",", ":"), allow_nan=False))
    merged_config = copy.deepcopy(configs[0])
    merged_config["frame_shard"] = None
    merged_config["save_results"] = str(out)
    out.with_suffix(".config.yaml").write_text(yaml.safe_dump(merged_config, sort_keys=False))

    s = merged["summary"]
    print(f"Merged {s['num_shards']} shards -> {out} ({s['num_samples']} frames, "
          f"ADD AUC@100mm={s['add_auc_100mm']:.4f}, mean={s['add_mean_mm']:.2f} mm)")


if __name__ == "__main__":
    main()
