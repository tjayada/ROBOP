import numpy as np
import pytest

from analysis.analysis_utils import add_auc as analysis_add_auc
from evaluation.eval_utils import compute_add_metrics


def test_add_metrics_known_reference():
    errors = np.array([0.0, 50.0, 1000.0])
    metrics = compute_add_metrics(errors)
    assert metrics == pytest.approx({
        "add_mean_mm": 350.0,
        "add_median_mm": 50.0,
        "add_p90_mm": 810.0,
        "add_p95_mm": 905.0,
        "add_auc_100mm": 0.49995,
        "add_auc_400mm": 0.6249875,
        "add_at_100mm": 2.0 / 3.0,
        "add_at_50mm": 1.0 / 3.0,
    })
    assert analysis_add_auc(errors, 100.0) == pytest.approx(0.49995)
    assert analysis_add_auc(errors, 400.0) == pytest.approx(0.6249875)
