"""Sharded eval must cover the dataset exactly once and merge to single-run numbers.

The 32k panda-orb run is split across GPUs with frame_shard=k/M and recombined by
evaluation/merge_shards.py. A dropped or duplicated frame would silently move the
ADD metric, so these pin the two invariants that prevent it: the M shards tile the
selection with no gaps or overlaps, and merging recomputes the same ADD as one job.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Mirror how the eval scripts run: evaluation/ on the path, imported as top-level.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from eval_utils import compute_add_metrics, select_eval_frame_indices  # noqa: E402
from merge_shards import _normalized_config, merge_shard_results  # noqa: E402


@pytest.mark.parametrize(
    "n_total, num_eval, m",
    [(32315, None, 64), (32315, 1000, 64)],
)
def test_shards_tile_selection(n_total, num_eval, m):
    full = select_eval_frame_indices(n_total, num_eval)
    shards = [select_eval_frame_indices(n_total, num_eval, None, f"{k}/{m}") for k in range(m)]
    # Concatenated in k-order the shards reproduce the selection: disjoint, complete,
    # order-preserving all at once.
    assert [i for sh in shards for i in sh] == full
    sizes = [len(sh) for sh in shards]
    assert max(sizes) - min(sizes) <= 1  # array_split keeps blocks near-equal


def _query(frame_id, add_m, frame_time_ms=None):
    diag = {"frame_time_ms": frame_time_ms} if frame_time_ms is not None else None
    return {"frame_id": frame_id, "add_m": add_m, "add_m_mm": round(add_m * 1000, 3),
            "pnp_failed": add_m >= 1.0, "diagnostics": diag}


def _shard(queries):
    return {"summary": {"dataset": "panda_orb", "num_params": 123,
                        "checkpoint_sha256": "weights", "runtime_median_ms": 99.0},
            "detections": {"path": "x", "sha256": "detections"},
            "provenance": {"timestamp_utc": "ignored", "git": "abc"},
            "queries": queries}


def test_merge_recomputes_single_run_add_metrics():
    rng = np.random.default_rng(0)
    errs = list(rng.uniform(0.0, 0.3, size=50))  # metres
    errs[7] = 1.0  # PnP-failure sentinel -> 1000 mm
    queries = [_query(f"{i:06d}", e, frame_time_ms=float(i)) for i, e in enumerate(errs)]

    ref = compute_add_metrics(np.array(errs) * 1000.0)  # one-job reference

    blocks = np.array_split(np.arange(50), 3)
    merged = merge_shard_results([_shard([queries[i] for i in blk]) for blk in blocks])

    s = merged["summary"]
    for key, val in ref.items():
        assert s[key] == pytest.approx(val)
    assert s["num_samples"] == 50
    assert s["num_shards"] == 3
    assert s["dataset"] == "panda_orb"      # base summary preserved
    assert s["num_params"] == 123           # model_stats preserved
    # runtime recomputed over the union: median of frames 5..49 (global warm-up skip)
    assert s["runtime_median_ms"] == float(np.median(np.arange(5, 50, dtype=float)))


def test_merge_ignores_shard_local_summary_differences():
    # shard-local keys (runtime) may differ: must not block the merge or leak through
    left = _shard([_query(f"{i:06d}", 0.1, frame_time_ms=float(i)) for i in range(3)])
    right = _shard([_query(f"{i:06d}", 0.1, frame_time_ms=float(i)) for i in range(3, 6)])
    right["summary"]["runtime_median_ms"] = 42.0
    merged = merge_shard_results([left, right])
    assert merged["summary"]["runtime_median_ms"] == 5.0  # union frame 5, post warm-up skip


def test_merge_rejects_overlapping_shards():
    q = [_query("000001", 0.1)]
    with pytest.raises(ValueError):
        merge_shard_results([_shard(q), _shard(q)])


def test_merge_rejects_mixed_run_identity():
    changes = (
        lambda shard: shard["detections"].update(sha256="other"),
        lambda shard: shard["provenance"].update(git="other"),
        lambda shard: shard["summary"].update(checkpoint_sha256="other"),
    )
    for change in changes:
        left = _shard([_query("000001", 0.1)])
        right = _shard([_query("000002", 0.2)])
        change(right)
        with pytest.raises(ValueError):
            merge_shard_results([left, right])


def test_config_identity_ignores_only_shard_output_fields():
    left = {"estimator": {"type": "nemo", "num_views": 32},
            "frame_shard": "0/2", "save_results": "shard_0/results.json"}
    right = {"estimator": {"type": "nemo", "num_views": 32},
             "frame_shard": "1/2", "save_results": "shard_1/results.json"}
    assert _normalized_config(left) == _normalized_config(right)

    right["estimator"]["num_views"] = 16
    assert _normalized_config(left) != _normalized_config(right)
