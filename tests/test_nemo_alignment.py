import math

import pytest
import torch

na = pytest.importorskip("robop.nemo_estimator")


def _pose(R, t):
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _random_poses(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    poses = []
    for _ in range(n):
        A = torch.randn(3, 3, generator=g)
        Q, R = torch.linalg.qr(A)
        Q = Q * torch.sign(torch.diagonal(R))
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        t = torch.tensor([0.0, 0.0, 1.0]) + 0.1 * torch.randn(3, generator=g)
        poses.append(_pose(Q, t))
    return torch.stack(poses)


def test_rotation_error_normalized():
    R = _rot_z(math.pi / 2).unsqueeze(0)
    eye = torch.eye(3).unsqueeze(0)
    # the cosine is clamped to 1-1e-6, so a zero error bottoms out near 4.5e-4
    assert na.rotation_error_normalized(eye, eye)[0] < 1e-3
    assert torch.isclose(na.rotation_error_normalized(R, eye)[0], torch.tensor(0.5), atol=1e-3)


def _fit_rotation(target, differentiable_axis, steps=300):
    """Optimize an axis-angle vector towards a target rotation, starting from
    the same initialization the alignment uses."""
    r = torch.nn.Parameter(torch.ones(3) * 1e-2)
    opt = torch.optim.Adam([r], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((na.axis_angle_to_matrix(r, differentiable_axis=differentiable_axis)
                 - target) ** 2).sum()
        loss.backward()
        opt.step()
    return r.detach()


def test_axis_moves_only_when_requested():
    """The reference rebuilds the skew matrix as a constant, so the optimizer
    can only change the angle about the initial axis; the differentiable
    variant can also turn the axis toward the target."""
    target = _rot_z(0.6)
    init_axis = torch.ones(3) / math.sqrt(3.0)

    r_ref = _fit_rotation(target, differentiable_axis=False)
    cos_ref = torch.nn.functional.cosine_similarity(
        r_ref.unsqueeze(0), init_axis.unsqueeze(0)).abs().item()
    assert cos_ref > 0.999    # axis unchanged

    r_free = _fit_rotation(target, differentiable_axis=True)
    cos_free = torch.nn.functional.cosine_similarity(
        r_free.unsqueeze(0), init_axis.unsqueeze(0)).abs().item()
    assert cos_free < 0.9     # axis turned toward the target's z axis


def test_recovers_scale_and_offset():
    """Estimates built from ground truth with a known scale and offset must
    return those parameters. The model is gt = s * (est_R @ t + est_t), so the
    estimates are built by inverting exactly that."""
    gt = _random_poses(8, seed=1)
    scale = 2.5
    offset = torch.tensor([0.012, -0.02, 0.008])
    est = gt.clone()
    est[:, :3, 3] = gt[:, :3, 3] / scale - gt[:, :3, :3] @ offset

    out = na.optimize_similarity(gt, est, k_best=5, max_iter=1500, lr=0.05)
    assert out["scale"].item() == pytest.approx(scale, rel=0.02)
    assert torch.allclose(out["center_offset"], offset, atol=5e-3)
    assert max(out["final_translation_errors"]) < 5e-3


def test_keeps_the_k_best_templates():
    """Templates with the worst initial rotation error must be excluded."""
    gt = _random_poses(8, seed=2)
    est = gt.clone()
    for bad in (0, 3):
        est[bad, :3, :3] = _rot_z(1.2) @ est[bad, :3, :3]

    out = na.optimize_similarity(gt, est, k_best=5, max_iter=50)
    assert len(out["kept_template_ids"]) == 5
    assert 0 not in out["kept_template_ids"] and 3 not in out["kept_template_ids"]
