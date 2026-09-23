import json

import numpy as np
import torch
from omegaconf import OmegaConf

from evaluation.eval_utils import (
    EstimatorAdapter,
    pose_output_problems,
    save_results_json,
)


def _valid_pose():
    T = np.eye(4)
    T[2, 3] = 1.5  # base origin 1.5 m in front of the camera
    return T


def test_valid_pose_passes():
    assert pose_output_problems(_valid_pose()) == []
    assert pose_output_problems(torch.from_numpy(_valid_pose()).float()) == []


def test_behind_camera_rejected():
    T = _valid_pose()
    T[2, 3] = -0.5
    assert any("front of the camera" in p for p in pose_output_problems(T))


class _InvalidPoseEstimator:
    INPUT_CONTRACT = {"mask": False, "bbox_xyxy": False, "frame_key": False,
                      "mask_3x3_opening": False}

    def __init__(self, pose):
        self._pose = pose
        self._last_frame_diagnostics = {}

    def inference_single_image(self, img, joint_angles, K=None, **kwargs):
        return self._pose, None, None, None, None


def test_adapter_converts_invalid_pose_to_failure():
    T = _valid_pose()
    T[0, 3] = np.inf
    est = _InvalidPoseEstimator(torch.from_numpy(T))
    result = EstimatorAdapter(est, torch.eye(3)).inference_single_image(
        "img", torch.zeros(7), frame_key="frame0")
    assert result[0] is None
    assert "non-finite" in est._last_frame_diagnostics["invalid_pose_reason"]


def test_save_results_json_is_strict(tmp_path):
    payload = {
        "errors_mm": [12.5, np.nan, np.float32(7.25), np.inf],
        "count": np.int64(3),
        "flag": np.bool_(True),
        "pose": np.eye(2),
    }
    out = tmp_path / "results.json"
    save_results_json(out, payload, OmegaConf.create({"a": 1}))
    text = out.read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["errors_mm"] == [12.5, None, 7.25, None]
    assert data["count"] == 3 and data["flag"] is True
    assert data["pose"] == [[1.0, 0.0], [0.0, 1.0]]
    assert out.with_suffix(".config.yaml").exists()
