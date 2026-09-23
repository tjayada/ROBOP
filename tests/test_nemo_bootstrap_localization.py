"""Bootstrap localization picks one connected component, not the extent of
every foreground island."""
import numpy as np
import pytest
import torch

robop_nemo = pytest.importorskip("robop.nemo_estimator")
NeMORobotPoseEstimator = robop_nemo.NeMORobotPoseEstimator


class _Localizer:
    """Exercises _decoder_mask_bbox_orig without a model or renderer."""

    def __init__(self, min_pixels=64, pad=0.0, mask_threshold=0.5):
        self.config = type("cfg", (), {
            "bootstrap_min_mask_pixels": min_pixels,
            "bootstrap_mask_pad": pad,
            "mask_threshold": mask_threshold,
            "conf_threshold": 0.1,
        })()
        self._last_bootstrap_components = None

    _decoder_mask_bbox_orig = NeMORobotPoseEstimator._decoder_mask_bbox_orig


def _decoder(mask):
    return {"mask": torch.from_numpy(mask).float()[None, None]}


def _blob(mask, x0, y0, w, h):
    mask[y0:y0 + h, x0:x0 + w] = 1.0
    return mask


def test_selects_the_largest_component():
    """A small far-away island must not stretch the box across the frame."""
    m = np.zeros((64, 64), dtype=np.float32)
    _blob(m, 4, 4, 20, 20)      # the arm: 400 px
    _blob(m, 60, 60, 3, 3)      # a stray island: 9 px
    loc = _Localizer(min_pixels=64)
    loc._last_crop_params = (0, 0, 64, 64)

    box = loc._decoder_mask_bbox_orig(_decoder(m))
    assert box is not None
    np.testing.assert_allclose(box, [4, 4, 24, 24], atol=1e-5)
    assert loc._last_bootstrap_components["n_components"] == 2
    assert loc._last_bootstrap_components["selected_area"] == 400


def test_floor_applies_to_the_component_not_the_sum():
    """Several small islands must not pass the floor by adding up."""
    m = np.zeros((64, 64), dtype=np.float32)
    for x in (2, 12, 22, 32, 42):
        _blob(m, x, 5, 4, 4)     # 5 islands of 16 px, 80 px in total
    loc = _Localizer(min_pixels=64)
    loc._last_crop_params = (0, 0, 64, 64)

    assert loc._decoder_mask_bbox_orig(_decoder(m)) is None
    assert loc._last_bootstrap_components["selected_area"] == 16


def test_box_is_mapped_back_through_the_crop():
    m = _blob(np.zeros((32, 32), dtype=np.float32), 8, 8, 16, 16)
    loc = _Localizer(min_pixels=16)
    loc._last_crop_params = (100, 50, 320, 320)   # crop offset and scale

    box = loc._decoder_mask_bbox_orig(_decoder(m))
    # decoder grid 32 px spans a 320 px crop starting at (100, 50): scale 10
    np.testing.assert_allclose(box, [180, 130, 340, 290], atol=1e-4)


@pytest.mark.parametrize("h,w,n", [(480, 640, 2), (720, 1280, 4)])
def test_patch_scan_covers_the_whole_frame(h, w, n):
    """A gap in the scan silently loses arms that fall in it (16:9 frames put the
    arm outside the centre square in up to 4% of Hydra frames)."""
    origins = NeMORobotPoseEstimator._bootstrap_patch_origins(None, h, w)
    s = min(h, w)
    assert len(origins) == n
    covered = np.zeros((h, w), dtype=bool)
    for x, y in origins:
        assert 0 <= x <= w - s and 0 <= y <= h - s      # patch stays inside the frame
        covered[y:y + s, x:x + s] = True
    assert covered.all()


def test_falls_back_to_confidence_when_no_mask_head():
    conf = np.zeros((64, 64), dtype=np.float32)
    conf[10:30, 10:30] = 0.9
    loc = _Localizer(min_pixels=64)
    loc._last_crop_params = (0, 0, 64, 64)

    box = loc._decoder_mask_bbox_orig({"conf": torch.from_numpy(conf)[None, None]})
    np.testing.assert_allclose(box, [10, 10, 30, 30], atol=1e-5)
