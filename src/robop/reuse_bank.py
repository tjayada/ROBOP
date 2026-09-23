"""Config-keyed representation bank for NeMO (online lazy policy).

Arrival-order cache: a query config reuses a stored representation when some bank
entry is within `tolerance_mm` of it in d_surf, otherwise it is encoded and added.
Bank entries end up pairwise > tolerance apart, so the bank is no larger than the
optimal tolerance/2 cover (packing-covering inequality).

The d_surf metric mirrors `RobotGeometry` in analysis/joint_reuse/analyze_perturbation.py -
per-link bounding-box corners posed by FK, averaged over the links that moved. It
has to stay identical to that one, because the tolerance is read straight off the
joint-perturbation delta* and the covering analysis.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

MOVING_EPS_M = 1e-9   # analysis/joint_reuse/analyze_perturbation.py
_N_CORNERS = 8


def _dsurf(bank_clouds: np.ndarray, W: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """d_surf (mm) from the ref cloud to each bank cloud: mean over MOVING links."""
    disp = np.linalg.norm(bank_clouds - ref[None], axis=2)   # (K, M)
    link_mean = disp @ W                                     # (K, n_links)
    moving = link_mean > MOVING_EPS_M
    den = moving.sum(1)
    num = (link_mean * moving).sum(1)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0) * 1000.0


class _Geometry:
    """Per-link local bbox corners (loaded once) + FK, giving one probe cloud per config."""

    def __init__(self, robot_name: str, robot_kin):
        import trimesh
        from robot_renderer import registry

        registry._register_builtins()
        entry = registry.get_robot_entry(robot_name)
        self.kin = robot_kin
        self.corners_local = []
        for files in entry["mesh_files"]:
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for f in files:
                m = trimesh.load(f, force="mesh", process=False)
                lo = np.minimum(lo, m.bounds[0])
                hi = np.maximum(hi, m.bounds[1])
            self.corners_local.append(np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                 for z in (lo[2], hi[2])]))
        self.n_links = len(self.corners_local)
        # Point -> link mean-reduction matrix, so a cloud distance reduces per link.
        self.W = np.zeros((self.n_links * _N_CORNERS, self.n_links))
        for i in range(self.n_links):
            self.W[i * _N_CORNERS:(i + 1) * _N_CORNERS, i] = 1.0 / _N_CORNERS

    def cloud(self, q) -> np.ndarray:
        """Probe points for one config, in the robot base frame: (n_links*8, 3)."""
        R, t = self.kin.get_joint_R_t(np.asarray(q, dtype=np.float64))
        return np.concatenate(
            [c @ R[i].T + t[i] for i, c in enumerate(self.corners_local)]
        )


class ReuseBank:
    """Stores (config, representation) pairs and serves the nearest within tolerance."""

    def __init__(self, robot_name: str, robot_kin, tolerance_mm: float):
        self.tolerance_mm = float(tolerance_mm)
        self.geom = _Geometry(robot_name, robot_kin)
        self._clouds: list[np.ndarray] = []
        self._payloads: list[Any] = []
        self._stack: Optional[np.ndarray] = None   # cached np.stack(self._clouds)
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._payloads)

    def lookup(self, q) -> Tuple[Optional[Any], float, Optional[int]]:
        """Nearest stored representation within tolerance: (payload, d_surf mm, index).

        Payload and index are None on a miss; distance is NaN while the bank is empty.
        """
        if not self._payloads:
            return None, float("nan"), None
        if self._stack is None:
            self._stack = np.stack(self._clouds)
        d = _dsurf(self._stack, self.geom.W, self.geom.cloud(q))
        i = int(d.argmin())
        if d[i] <= self.tolerance_mm:
            self.hits += 1
            return self._payloads[i], float(d[i]), i
        return None, float(d[i]), None

    def insert(self, q, payload: Any) -> None:
        self._clouds.append(self.geom.cloud(q))
        self._payloads.append(payload)
        self._stack = None
        self.misses += 1
