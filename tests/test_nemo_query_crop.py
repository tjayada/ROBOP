"""The NeMO query crop must stay square so the arm is never stretched into
the square network input.

_load_query_image always finishes with the resize to nemo_encoding_size (the
published encoding regime), so the no-stretch guarantee lives entirely in the
crop geometry: the crop recorded in _last_crop_params has to be square BEFORE
that resize. These tests assert exactly that.
"""
import pytest
import torch

robop_nemo = pytest.importorskip("robop.nemo_estimator")
NeMORobotPoseEstimator = robop_nemo.NeMORobotPoseEstimator


class _Cropper:
    """Exercises _load_query_image without building a model or renderer."""

    def __init__(self, bbox_crop_pad=0.0):
        self.config = type("cfg", (), {"bbox_crop_pad": bbox_crop_pad})()
        self._last_crop_params = None

    _load_query_image = NeMORobotPoseEstimator._load_query_image


def _crop(image_hw, bbox, pad=0.0, size=224):
    h, w = image_hw
    img = torch.rand(1, 3, h, w)
    c = _Cropper(bbox_crop_pad=pad)
    out = c._load_query_image(
        img, nemo_encoding_size=size,
        device=torch.device("cpu"), crop_mode="bbox", bbox_xyxy=bbox,
    )
    return out, c._last_crop_params


def test_crop_is_square_for_an_interior_box():
    out, (x, y, cw, ch) = _crop((720, 1280), [500.0, 300.0, 700.0, 500.0])
    assert cw == ch
    assert out.shape[-2:] == (224, 224)   # square crop -> undistorted encoder input


def test_oversized_box_shrinks_instead_of_stretching():
    """A requested square larger than the image height must stay square."""
    out, (x, y, cw, ch) = _crop((720, 1280), [100.0, 10.0, 1200.0, 710.0])
    assert cw == ch == 720          # clamped to the short side, still square
    assert 0 <= x and x + cw <= 1280
    assert 0 <= y and y + ch <= 720


def test_box_at_the_image_edge_slides_inside():
    out, (x, y, cw, ch) = _crop((720, 1280), [1150.0, 600.0, 1279.0, 719.0])
    assert cw == ch
    assert x + cw <= 1280 and y + ch <= 720
    assert x >= 0 and y >= 0


def test_bbox_mode_without_a_detection_raises():
    """crop_mode='bbox' must not quietly become a center crop: the run would
    then be named after a regime it did not use."""
    c = _Cropper()
    with pytest.raises(ValueError, match="needs a detection bbox"):
        c._load_query_image(
            torch.rand(1, 3, 480, 640),
            nemo_encoding_size=224, device=torch.device("cpu"),
            crop_mode="bbox", bbox_xyxy=None,
        )
