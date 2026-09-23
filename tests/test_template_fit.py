import json

import numpy as np
import pytest

import estimator_utils

# GigaPose's fixed template camera, the first consumer of these helpers.
K = np.array([[572.4114, 0.0, 320.0], [0.0, 573.57043, 240.0], [0.0, 0.0, 1.0]])
W, H = 640, 480


def _random_rotations(n, seed):
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(n):
        A = rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        rotations.append(Q)
    return rotations


def test_bounding_radius():
    v = np.array([[1.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 0.0, 2.0]])
    assert estimator_utils.bounding_radius_mm(v) == 3.0
    with pytest.raises(ValueError):
        estimator_utils.bounding_radius_mm(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        estimator_utils.bounding_radius_mm(np.zeros((4, 2)))


def test_distance_formula_is_the_exact_sphere_bound():
    r = 500.0
    margin = 0.05
    d = estimator_utils.viewsphere_distance_mm(r, K, W, H, margin=margin)
    # The projected silhouette radius of the sphere at distance d must equal
    # (1 - margin) * available half extent on the binding axis and not exceed
    # it on the other.
    proj = lambda f: f * r / np.sqrt(d * d - r * r)  # noqa: E731
    ratio_x = proj(K[0, 0]) / ((1.0 - margin) * min(K[0, 2], (W - 1) - K[0, 2]))
    ratio_y = proj(K[1, 1]) / ((1.0 - margin) * min(K[1, 2], (H - 1) - K[1, 2]))
    assert max(ratio_x, ratio_y) == pytest.approx(1.0, abs=1e-12)
    assert min(ratio_x, ratio_y) <= 1.0


def test_points_in_sphere_project_inside_for_every_rotation():
    rng = np.random.default_rng(0)
    r = 450.0
    pts = rng.normal(size=(2000, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * (r * rng.uniform(0, 1, size=(2000, 1)))
    d = estimator_utils.viewsphere_distance_mm(r, K, W, H, margin=0.05)
    for R in _random_rotations(25, seed=1):
        cam = pts @ R.T + np.array([0.0, 0.0, d])
        assert cam[:, 2].min() > 0
        uv = cam @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        assert uv[:, 0].min() >= 0 and uv[:, 0].max() <= W - 1
        assert uv[:, 1].min() >= 0 and uv[:, 1].max() <= H - 1


def test_square_camera_from_K_upstream_construction():
    K_data = np.array([[911.3, 0.0, 643.84], [0.0, 910.96, 368.39], [0.0, 0.0, 1.0]])
    K_sq, side = estimator_utils.square_camera_from_K(K_data, 1280, 720, patch_size=14)
    assert side == 14 * int(1280 / 14)
    assert K_sq[0, 0] == K_data[0, 0] and K_sq[1, 1] == K_data[1, 1]
    assert K_sq[0, 2] == pytest.approx(K_data[0, 2] - 0.5 * (1280 - side))
    assert K_sq[1, 2] == pytest.approx(K_data[1, 2] - 0.5 * (720 - side))


def test_check_mask_fit_interior_blob():
    m = np.zeros((20, 30))
    m[5:10, 7:12] = 1
    stats = estimator_utils.check_mask_fit(m, min_pixels=10)
    assert stats["n_pixels"] == 25
    assert stats["bbox_xyxy"] == [7, 5, 12, 10]  # max-exclusive
    assert not stats["touches_border"]
    assert stats["ok"]


def test_check_mask_fit_border_and_min_pixels():
    m = np.zeros((20, 20))
    m[0, 5] = 1  # touches top border
    m[5:8, 5:8] = 1
    stats = estimator_utils.check_mask_fit(m, min_pixels=1)
    assert stats["touches_border"]
    assert not stats["ok"]
    m2 = np.zeros((20, 20))
    m2[5, 5] = 1
    assert not estimator_utils.check_mask_fit(m2, min_pixels=2)["ok"]
    with pytest.raises(ValueError):
        estimator_utils.check_mask_fit(np.zeros((2, 2, 2)), min_pixels=1)


def _manifest_fixture(tmp_path, n=3):
    meta = {"estimator": "gigapose", "n_views": n, "distance_mm": 1234.5}
    views = []
    for i in range(n):
        name = f"{i:06d}.png"
        (tmp_path / name).write_bytes(b"x")
        views.append({"file": name, "n_pixels": 500, "bbox_xyxy": [1, 1, 5, 5],
                      "touches_border": False, "ok": True})
    estimator_utils.write_template_manifest(tmp_path, meta, views)
    return meta, views


def test_manifest_roundtrip_and_missing(tmp_path):
    assert estimator_utils.load_template_manifest(tmp_path) is None
    meta, views = _manifest_fixture(tmp_path)
    manifest = estimator_utils.load_template_manifest(tmp_path)
    assert manifest["meta"] == meta
    assert manifest["views"] == views


def test_manifest_problems_valid(tmp_path):
    _manifest_fixture(tmp_path)
    manifest = estimator_utils.load_template_manifest(tmp_path)
    assert estimator_utils.manifest_problems(manifest, tmp_path) == []


def test_manifest_problems_detects_defects(tmp_path):
    _, views = _manifest_fixture(tmp_path)
    manifest = estimator_utils.load_template_manifest(tmp_path)

    (tmp_path / views[1]["file"]).unlink()
    problems = estimator_utils.manifest_problems(manifest, tmp_path)
    assert any("file missing" in p for p in problems)

    manifest2 = json.loads(json.dumps(manifest))
    manifest2["views"][0]["ok"] = False
    problems = estimator_utils.manifest_problems(manifest2, tmp_path)
    assert any("failed fit checks" in p for p in problems)


def test_manifest_problems_detects_truncated_view_list(tmp_path):
    """A view list shorter than the count the manifest declares = incomplete render."""
    _manifest_fixture(tmp_path)
    manifest = estimator_utils.load_template_manifest(tmp_path)
    manifest["views"].pop()
    problems = estimator_utils.manifest_problems(manifest, tmp_path)
    assert any("expected 3" in p for p in problems)
