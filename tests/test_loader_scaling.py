"""Intrinsics must follow the pixels a resize actually produces."""
import numpy as np
import pytest

from evaluation.loaders.utils import scale_K_to_size, scaled_size

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])


def test_scaled_size_rounds():
    assert scaled_size(1280, 720, 1.0) == (1280, 720)
    assert scaled_size(1280, 720, 0.5) == (640, 360)
    assert scaled_size(1280, 720, 0.333) == (426, 240)


def test_non_integral_scale_uses_separate_axis_factors():
    """1280x720 at 0.333 rounds to 426x240, so x and y differ: 426/1280 is not
    240/720. Scaling both axes by the nominal 0.333 would misplace the
    principal point."""
    out_w, out_h = scaled_size(1280, 720, 0.333)
    out = scale_K_to_size(K, 1280, 720, out_w, out_h)
    sx, sy = out_w / 1280, out_h / 720
    assert sx != sy
    assert out[0, 0] == pytest.approx(900.0 * sx)
    assert out[1, 1] == pytest.approx(900.0 * sy)
    assert out[0, 2] == pytest.approx(640.0 * sx)
    assert out[1, 2] == pytest.approx(360.0 * sy)
    # the nominal factor would have been wrong on at least one axis
    assert out[1, 1] != pytest.approx(900.0 * 0.333)


def test_principal_point_stays_at_the_image_centre():
    """A centred principal point must stay centred after scaling."""
    out_w, out_h = scaled_size(1280, 720, 0.4)
    out = scale_K_to_size(K, 1280, 720, out_w, out_h)
    assert out[0, 2] == pytest.approx(out_w / 2, abs=0.5)
    assert out[1, 2] == pytest.approx(out_h / 2, abs=0.5)
