"""Deterministic joint-space perturbations for the noise analysis.

Each frame gets a stable unit direction derived from its key and the run seed.
The requested L2 magnitude is clamped to the robot's joint limits.
"""
from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

# Registry name -> joint count of the q vector the eval scripts pass (adapter input).
EVAL_NUM_JOINTS = {
    "panda": 7,
    "baxter_left_arm": 7,
    "owi535": 4,
    "lbr_med7": 7,
    "xarm7": 7,
    "meca500": 6,
}


def _frame_entropy(frame_key: str) -> int:
    """Stable 128-bit integer from a frame identifier (image path / meas-key)."""
    return int.from_bytes(hashlib.sha256(frame_key.encode("utf-8")).digest()[:16], "little")


def make_rng(global_seed: int, frame_key: str) -> np.random.Generator:
    """Dedicated per-frame Generator - independent of numpy's global RNG."""
    return np.random.default_rng(np.random.SeedSequence([int(global_seed), _frame_entropy(frame_key)]))


def sample_direction(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform direction on the unit sphere in R^n (normalized Gaussian)."""
    while True:
        u = rng.standard_normal(n)
        norm = float(np.linalg.norm(u))
        if norm > 1e-12:  # zero-vector guard; practically never loops
            return u / norm


def resolve_qlim(robot_kin, n_joints: int) -> np.ndarray:
    """Return joint limits aligned with the evaluation joint vector.

    Baxter uses the final seven columns of its 15-DOF model; Panda uses the
    first seven before its two gripper joints. Other adapters align directly.
    """
    if hasattr(robot_kin, "qlim"):
        qlim = np.asarray(robot_kin.qlim, dtype=np.float64)
    else:
        robot = getattr(robot_kin, "_robot", None)
        if robot is None:
            raise ValueError(
                f"Cannot resolve joint limits: {type(robot_kin).__name__} has neither "
                f".qlim nor ._robot (roboticstoolbox ERobot)."
            )
        qlim = np.asarray(robot.qlim, dtype=np.float64)
        if qlim.shape != (2, n_joints):
            if type(robot_kin).__name__ == "BaxterAdapter" and qlim.shape[1] >= n_joints:
                # BaxterAdapter: q_full[-7:] = q -> eval q maps to the LAST n_joints joints.
                qlim = qlim[:, -n_joints:]
            elif type(robot_kin).__name__ == "PandaAdapter" and qlim.shape[1] >= n_joints:
                # Panda URDF loads 9 actuated joints (7 arm + 2 gripper fingers), but
                # PANDA_LINK_INDICES excludes the finger chain and production fkine(q)
                # succeeds with a 7-vector - proof the arm occupies jindex 0..6 (fkine
                # indexes q by jindex). Eval q therefore maps to the FIRST n_joints.
                qlim = qlim[:, :n_joints]
            else:
                raise ValueError(
                    f"Joint-limit misalignment for {type(robot_kin).__name__}: "
                    f"ERobot qlim shape {qlim.shape}, eval q has {n_joints} joints, "
                    f"and no known padding rule applies. Refusing to guess."
                )
    if qlim.shape != (2, n_joints):
        raise ValueError(f"qlim shape {qlim.shape} != (2, {n_joints}).")
    if not np.all(np.isfinite(qlim)):
        bad = np.where(~np.all(np.isfinite(qlim), axis=0))[0].tolist()
        raise ValueError(
            f"Non-finite joint limits at joint indices {bad} for {type(robot_kin).__name__} "
            f"- URDF likely missing <limit> tags; fix before running the perturbation sweep."
        )
    if not np.all(qlim[0] < qlim[1]):
        bad = np.where(qlim[0] >= qlim[1])[0].tolist()
        raise ValueError(
            f"Degenerate joint limits (lower >= upper) at joint indices {bad} "
            f"for {type(robot_kin).__name__}: {qlim[:, bad].tolist()}"
        )
    return qlim


def perturb(
    q: np.ndarray,
    magnitude_deg: float,
    global_seed: int,
    frame_key: str,
    qlim: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """
    q̃ = clip(q + deg2rad(magnitude_deg)·u, qlim), u uniform on the unit sphere,
    deterministic per (global_seed, frame_key).

    Returns (q_tilde float64 (n,), record dict - JSON-serializable):
      requested_deg      the commanded L2 magnitude (deg)
      direction          u (list, unit L2 norm)
      dq_rad             realized q̃ − q AFTER clamping (list, rad)
      realized_norm_deg  ‖dq_rad‖ in degrees (== requested_deg unless clamped)
      clamp_count        number of clamped joints
      clamped_joints     indices of clamped joints
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = q.shape[0]
    rng = make_rng(global_seed, frame_key)
    u = sample_direction(rng, n)
    dq_requested = np.deg2rad(float(magnitude_deg)) * u
    q_tilde_raw = q + dq_requested
    q_tilde = np.clip(q_tilde_raw, qlim[0], qlim[1])
    clamped = q_tilde != q_tilde_raw
    dq_realized = q_tilde - q
    record = {
        "requested_deg": float(magnitude_deg),
        "direction": u.tolist(),
        "dq_rad": dq_realized.tolist(),
        "realized_norm_deg": float(np.rad2deg(np.linalg.norm(dq_realized))),
        "clamp_count": int(clamped.sum()),
        "clamped_joints": np.where(clamped)[0].astype(int).tolist(),
    }
    return q_tilde, record


# Self-test + joint-limit check

def _self_test() -> None:
    wide = np.array([[-10.0] * 7, [10.0] * 7])
    q = np.linspace(-0.5, 0.5, 7)

    # 1. Determinism: same (seed, frame_key) -> identical perturbation.
    qa, ra = perturb(q, 2.0, 12345, "frames/000042.png", wide)
    qb, rb = perturb(q, 2.0, 12345, "frames/000042.png", wide)
    assert np.array_equal(qa, qb) and ra == rb, "determinism broken"

    # 2. Frame / seed separation: different key or seed -> different direction.
    _, rc = perturb(q, 2.0, 12345, "frames/000043.png", wide)
    _, rd = perturb(q, 2.0, 54321, "frames/000042.png", wide)
    assert rc["direction"] != ra["direction"] != rd["direction"], "seeding not separating"

    # 3. Same direction across magnitudes (paired-ray design) + unit norm + exact L2.
    _, r_small = perturb(q, 0.1, 12345, "frames/000042.png", wide)
    da, ds = np.array(ra["direction"]), np.array(r_small["direction"])
    assert np.allclose(da, ds), "direction must be magnitude-independent per frame"
    assert abs(np.linalg.norm(da) - 1.0) < 1e-12, "direction not unit norm"
    assert abs(ra["realized_norm_deg"] - 2.0) < 1e-9, "unclamped realized norm != requested"
    assert ra["clamp_count"] == 0

    # 4. Clamping: tight limits -> flags set, realized norm < requested, q̃ within limits.
    tight = np.vstack([q - np.deg2rad(0.5), q + np.deg2rad(0.5)])
    qt, rt = perturb(q, 5.0, 12345, "frames/000042.png", tight)
    assert rt["clamp_count"] > 0 and rt["realized_norm_deg"] < 5.0, "clamp not registering"
    assert np.all(qt >= tight[0]) and np.all(qt <= tight[1]), "clamp violated limits"

    # 5. Global-RNG independence: perturb() must not consume np.random's stream.
    np.random.seed(0)
    expected = np.random.rand(3)
    np.random.seed(0)
    perturb(q, 2.0, 12345, "frames/000042.png", wide)
    assert np.array_equal(np.random.rand(3), expected), "global RNG state disturbed"

    # 6. Zero magnitude (adapter never calls this path, but must be exact if it did).
    qz, rz = perturb(q, 0.0, 12345, "frames/000042.png", wide)
    assert np.array_equal(qz, q) and rz["realized_norm_deg"] == 0.0

    # 7. Record is JSON-serializable (rides diagnostics -> results JSON).
    import json
    json.dumps(ra)

    print("joint_noise self-test: ALL PASS (7 checks)")


def _check_qlim() -> int:
    """Instantiate every registered robot adapter and validate resolve_qlim.

    Prints the limits used for perturbation clamping and returns a process exit code.
    """
    try:
        import robot_renderer  # noqa: F401  (installed in the ROBOP env)
        from robot_renderer import registry
    except ImportError:
        import sys
        from pathlib import Path
        rr_src = Path(__file__).resolve().parents[1] / "external" / "robot-renderer" / "src"
        if not rr_src.exists():
            print(f"FATAL: robot_renderer not importable and {rr_src} not found.")
            print("Run `git submodule update --init external/robot-renderer`.")
            return 2
        sys.path.insert(0, str(rr_src))
        from robot_renderer import registry

    registry._register_builtins()
    failures = 0
    for name, n_joints in EVAL_NUM_JOINTS.items():
        entry = registry._REGISTRY.get(name)
        if entry is None:
            print(f"{name:16s} SKIP - not in registry")
            continue
        try:
            robot_kin = entry["adapter_cls"](entry["urdf"])
            qlim = resolve_qlim(robot_kin, n_joints)
            spans = np.rad2deg(qlim[1] - qlim[0])
            print(f"{name:16s} OK   n={n_joints}  "
                  f"lo(deg)={np.round(np.rad2deg(qlim[0]), 1).tolist()}  "
                  f"hi(deg)={np.round(np.rad2deg(qlim[1]), 1).tolist()}  "
                  f"min_span={spans.min():.1f}°")
        except Exception as e:  # noqa: BLE001 - check must report every robot
            failures += 1
            print(f"{name:16s} FAIL - {e}")
    print("qlim check:", "ALL OK" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-qlim", action="store_true",
                        help="instantiate all registered robots and validate joint limits")
    args = parser.parse_args()
    if args.check_qlim:
        raise SystemExit(_check_qlim())
    _self_test()
