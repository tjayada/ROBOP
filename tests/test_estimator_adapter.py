import numpy as np
import pytest
import torch

from evaluation.eval_utils import EstimatorAdapter

K0 = torch.eye(3)


def _contract(**over):
    c = {"mask": True, "bbox_xyxy": True, "frame_key": False,
         "mask_3x3_opening": False}
    c.update(over)
    return c


class _Recorder:
    """Minimal estimator standing in for the real ones."""

    INPUT_CONTRACT = _contract()

    def __init__(self):
        self.calls = []

    def inference_single_image(self, img, joint_angles, K=None, mask=None,
                               bbox_xyxy=None, frame_key=None, **kwargs):
        self.calls.append({"K": K, "mask": mask, "bbox_xyxy": bbox_xyxy,
                           "frame_key": frame_key})
        return torch.eye(4), None, None, None, None


def test_inputs_routed_by_contract():
    est = _Recorder()
    EstimatorAdapter(est, K0).inference_single_image(
        "img", torch.zeros(7), mask="m", bbox_xyxy="b", frame_key="f")
    call = est.calls[0]
    assert call["mask"] == "m" and call["bbox_xyxy"] == "b"
    assert call["frame_key"] is None  # contract: estimator does not take it

    class NoDetections(_Recorder):
        INPUT_CONTRACT = _contract(mask=False, bbox_xyxy=False)

    est2 = NoDetections()
    EstimatorAdapter(est2, K0).inference_single_image(
        "img", torch.zeros(7), mask="m", bbox_xyxy="b")
    call2 = est2.calls[0]
    assert call2["mask"] is None and call2["bbox_xyxy"] is None


def test_frame_K_passed_through():
    est = _Recorder()
    K1 = np.eye(3) * 2.0
    EstimatorAdapter(est, K0).inference_single_image("img", torch.zeros(7), K=K1)
    got = est.calls[0]["K"]
    assert isinstance(got, torch.Tensor)
    assert torch.allclose(got, torch.as_tensor(K1, dtype=K0.dtype))


def test_default_K_when_none_given():
    est = _Recorder()
    EstimatorAdapter(est, K0).inference_single_image("img", torch.zeros(7))
    assert est.calls[0]["K"] is K0


def test_negative_joint_noise_magnitude_rejected():
    # A negative magnitude must raise, not silently run as an unlabeled baseline
    # (the gate is magnitude > 0.0, so a negative value would disable perturbation).
    with pytest.raises(ValueError):
        EstimatorAdapter(_Recorder(), K0,
                         joint_noise_cfg={"magnitude_deg": -1.0, "seed": 0})


def test_zero_joint_noise_magnitude_is_allowed():
    # 0.0 is the paired baseline (exact no-op), must construct fine.
    EstimatorAdapter(_Recorder(), K0,
                     joint_noise_cfg={"magnitude_deg": 0.0, "seed": 0})
