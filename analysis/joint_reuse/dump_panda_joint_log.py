"""Dump panda-orb NDDS joint states as an ordered ``(N, 7)`` numpy array.

Frame discovery and joint extraction match ``DREAMDataset`` without loading images.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump dense panda-orb joint log for covering analysis.")
    ap.add_argument("--data_folder", required=True, help="panda-orb NDDS folder (digit-prefixed .json)")
    ap.add_argument("--output", type=Path, default=Path("panda_orb_joint_log.npy"))
    ap.add_argument("--n_joints", type=int, default=7)
    args = ap.parse_args()

    folder = os.path.expanduser(args.data_folder)
    # Match DREAMDataset's sorted digit-prefixed frame order.
    names = sorted(f for f in os.listdir(folder) if f.endswith(".json") and f[0].isdigit())
    if not names:
        raise SystemExit(f"No digit-prefixed NDDS .json files in {folder}")

    configs = np.empty((len(names), args.n_joints), dtype=np.float64)
    for i, name in enumerate(names):
        with open(os.path.join(folder, name)) as f:
            joints = json.load(f)["sim_state"]["joints"]
        # radians, joint order as dream_loader.py:_load_sample.
        configs[i] = [joints[j]["position"] for j in range(args.n_joints)]

    np.save(args.output, configs)

    d = np.linalg.norm(np.diff(configs, axis=0), axis=1)  # consecutive-frame joint-L2 (rad)
    print(f"wrote {args.output}  shape={configs.shape}  (radians)")
    print(f"consecutive-frame joint-L2: median={np.degrees(np.median(d)):.2f} deg, "
          f"max={np.degrees(d.max()):.2f} deg")
    print("(smooth capture => small median; large spikes => segment cuts -- "
          "histogram these deltas to find segment boundaries)")


if __name__ == "__main__":
    main()
