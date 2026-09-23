"""A full-method run must report whether refinement actually ran."""
from evaluation.eval_utils import aggregate_frame_diagnostics


def _frames(n, **refinement):
    """n frames with the same refinement record."""
    return [{"refinement": dict(refinement)} for _ in range(n)]


def test_all_refined():
    out = aggregate_frame_diagnostics(
        _frames(20, enabled=True, attempted=True, succeeded=True, n_hypotheses=5))
    assert out["refiner_frames"] == 20      # refiner accounting has no warm-up skip
    assert out["refiner_failed"] == 0
    assert out["refiner_success_rate"] == 1.0


def test_early_failure_is_not_hidden_by_warmup():
    # A refiner failure in the first 5 frames must still be counted (the counters
    # are the gate against a full-method run silently going coarse).
    diags = (_frames(3, enabled=True, attempted=True, succeeded=False,
                     error="RuntimeError: boom")
             + _frames(17, enabled=True, attempted=True, succeeded=True))
    out = aggregate_frame_diagnostics(diags)
    assert out["refiner_frames"] == 20
    assert out["refiner_failed"] == 3
    assert out["refiner_success_rate"] < 1.0


def test_short_run_still_reports_refiner_counters():
    out = aggregate_frame_diagnostics(
        _frames(3, enabled=True, attempted=True, succeeded=True))
    assert out["refiner_frames"] == 3
    assert out["refiner_success_rate"] == 1.0


def test_coarse_only_runs_report_nothing():
    out = aggregate_frame_diagnostics(
        _frames(20, enabled=False, attempted=False, succeeded=False))
    assert "refiner_frames" not in out


def test_timing_phases_include_full_call_stages():
    diags = [{"timings_ms": {"coarse": 100.0, "postprocess": 5.0, "total": 140.0}}
             for _ in range(20)]
    out = aggregate_frame_diagnostics(diags)
    assert out["timing_coarse_median_ms"] == 100.0
    assert out["timing_postprocess_median_ms"] == 5.0
