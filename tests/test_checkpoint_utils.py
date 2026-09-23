import pytest
import torch

import estimator_utils


class _Net(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.fc = torch.nn.Linear(width, 2)


def test_matching_state_dict_passes():
    net = _Net()
    estimator_utils.assert_state_dict_matches(net, dict(net.state_dict()), "net")


def test_missing_key_raises():
    net = _Net()
    sd = dict(net.state_dict())
    del sd["fc.bias"]
    with pytest.raises(estimator_utils.CheckpointMismatch, match="missing"):
        estimator_utils.assert_state_dict_matches(net, sd, "net")


def test_unexpected_key_raises():
    net = _Net()
    sd = dict(net.state_dict())
    sd["extra.weight"] = torch.zeros(2)
    with pytest.raises(estimator_utils.CheckpointMismatch, match="unexpected"):
        estimator_utils.assert_state_dict_matches(net, sd, "net")


def test_shape_mismatch_raises():
    net = _Net(width=4)
    sd = dict(_Net(width=8).state_dict())
    with pytest.raises(estimator_utils.CheckpointMismatch, match="shape mismatch"):
        estimator_utils.assert_state_dict_matches(net, sd, "net")


def test_sha256_file(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"robop")
    digest = estimator_utils.sha256_file(p)
    assert len(digest) == 64
    assert digest == estimator_utils.sha256_file(p)
    (tmp_path / "other.bin").write_bytes(b"robop2")
    assert estimator_utils.sha256_file(tmp_path / "other.bin") != digest
