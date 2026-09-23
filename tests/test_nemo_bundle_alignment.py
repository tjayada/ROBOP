"""Bundle alignment requires updated features before decoding template images."""
import types

import pytest
import torch

robop_nemo = pytest.importorskip("robop.nemo_estimator")
NeMORobotConfig = robop_nemo.NeMORobotConfig
NeMORobotPoseEstimator = robop_nemo.NeMORobotPoseEstimator

_N_FEATS = 4


class _NemoStub:
    """Only the two model calls _setup_nemo_alignment can reach."""

    def point_encoder(self, points):
        # Depends on the input, so pre- and post-centering builds differ.
        return points.mean(dim=-1, keepdim=True).expand(*points.shape[:2], _N_FEATS)

    def decode_images(self, imgs, features):
        return None  # the monkeypatched PnP never inspects this


def _make_estimator(alignment_method, monkeypatch):
    """Estimator shell for _setup_nemo_alignment: no renderer, no weights."""
    est = object.__new__(NeMORobotPoseEstimator)
    est.config = NeMORobotConfig(alignment_method=alignment_method)
    est.device = torch.device("cpu")
    est.renderer = types.SimpleNamespace(
        config=types.SimpleNamespace(fill_frame=False, render_size=448),
        _K=torch.eye(3),
    )
    est.nemo_model = _NemoStub()
    est._cad_size_metric = 1.0
    est._cad_size_metric_aligned = 1.0
    est._cad_axis_extent_anchor = None
    est._cad_axis_extent_aligned = None
    est._mesh_surface_scale = None
    est._nemo_scale_factor = 1.0
    # The anchor PnP check runs on this path (fill_frame off); make its PnP
    # fail cheaply so the test stays dependency-light.
    monkeypatch.setattr(robop_nemo, "_pnp_pose_estimation_with_diagnostics",
                        lambda *a, **k: ([None], None))
    return est


def _nemo_dict(n=8):
    g = torch.Generator().manual_seed(0)
    return {
        "surface_points": torch.randn(1, n, 3, generator=g),
        "features_3d": torch.ones(1, n, _N_FEATS),
    }


def _templates_and_gt(n=3):
    return torch.rand(n, 3, 16, 16), [torch.eye(4) for _ in range(n)]


def test_bundle_fit_sees_updated_features(monkeypatch):
    est = _make_estimator("bundle", monkeypatch)
    nemo = _nemo_dict()
    templates, gt_poses = _templates_and_gt()
    seen = {}

    def _spy(nemo_arg, templates_arg, gt_poses_arg):
        seen["has_key"] = "features_3d_updated" in nemo_arg
        seen["value"] = nemo_arg.get("features_3d_updated")
        return None  # behave like a skipped fit: extent values are kept

    est._fit_bundle_alignment = _spy
    est._setup_nemo_alignment(nemo, templates, gt_poses, result=None)

    # Pre-fit build: features_3d + point_encoder of the UNCENTRED points.
    sp_orig = _nemo_dict()["surface_points"]
    expected_pre = 1.0 + sp_orig.mean(dim=-1, keepdim=True).expand(1, -1, _N_FEATS)
    assert seen["has_key"] is True
    assert torch.allclose(seen["value"], expected_pre)

    # After the call the points are centred and the features rebuilt on them.
    sp_final = nemo["surface_points"]
    centre = (sp_final.min(dim=1).values + sp_final.max(dim=1).values) / 2.0
    assert torch.allclose(centre, torch.zeros_like(centre), atol=1e-6)
    expected_post = 1.0 + sp_final.mean(dim=-1, keepdim=True).expand(1, -1, _N_FEATS)
    assert torch.allclose(nemo["features_3d_updated"], expected_post)
    assert not torch.allclose(nemo["features_3d_updated"], expected_pre)


def test_extent_path_does_not_call_the_fit(monkeypatch):
    est = _make_estimator("extent", monkeypatch)
    nemo = _nemo_dict()
    templates, gt_poses = _templates_and_gt()
    called = []

    est._fit_bundle_alignment = lambda *a: called.append(a)
    est._setup_nemo_alignment(nemo, templates, gt_poses, result=None)

    assert called == []
    assert "features_3d_updated" in nemo
