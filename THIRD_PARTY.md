# Third-party components

ROBOP's own code is MIT-licensed (see `LICENSE`). It integrates the following
third-party components, and with one exception none of them are vendored -
they are pulled in as git submodules, pip dependencies, or downloaded
checkpoints, and remain under their own licenses. The exception is a handful
of small constant tables copied into source; see "Vendored constants" below.

## Git submodules

A submodule stores only a URL and pinned commit; its contents are not
redistributed by this repository. The exact commits are listed here.

| Component | Pinned commit | License | Notes |
|---|---|---|---|
| [DLR-RM/NeMO](https://github.com/DLR-RM/NeMO) (`external/NeMO`) | `76478815baf2c99002cccbee98426773fa4e59cd` | MIT | Primary model (nemolib). Installed with `--no-deps`. Checkpoint ships as shards inside the upstream repo, which `make assemble-checkpoint` concatenates. |
| [facebookresearch/foundpose](https://github.com/facebookresearch/foundpose) (`external/foundpose`) | `3103473bfa3985fdf563bb31cefc0cae8f979ae0` | **CC-BY-NC-4.0** | WARNING: **Non-commercial license.** Using the FoundPose estimator restricts that pipeline to non-commercial use (fine for research). Path-injected at runtime, never vendored. |
| [nv-nguyen/gigapose](https://github.com/nv-nguyen/gigapose) (`external/gigapose`) | `17fcf97f493f79e56a215ab10ebff16d95cfe34b` | MIT | Also provides the MegaPose refiner code path (megapose6d, Apache-2.0). |
| [robot-renderer](https://github.com/tjayada/robot-renderer) (`external/robot-renderer`) | `e5a5d0feec358e3ad28f52ef9c00315067a8b381` | MIT | Template rendering + posed-mesh export. Installed editable with `--no-deps` (`make install`). Robot mesh/URDF assets inside it carry their own upstream licenses; see its `MESH_LICENSES/`. |

## Checkpoints & weights (downloaded, never vendored)

| Weights | Source | License |
|---|---|---|
| NeMO checkpoint | shards in the DLR-RM/NeMO repo (`make assemble-checkpoint`) | MIT (with the repo) |
| GigaPose `gigaPose_v1.ckpt` | [HuggingFace nv-nguyen/gigaPose](https://huggingface.co/datasets/nv-nguyen/gigaPose) (`make download-gigapose`) | MIT (with the repo) |
| MegaPose refiner models | [INRIA archive](https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models) (`make download-megapose`) | Apache-2.0 (megapose6d) |
| DINOv2 backbones | `torch.hub` (facebookresearch/dinov2), used by NeMO and FoundPose | Apache-2.0 |

## Vendored constants

`evaluation/loaders/craves_loader.py` copies the constant tables that define
the CRAVES 17-keypoint protocol, from
[ylabbe/robopose](https://github.com/ylabbe/robopose) (MIT, © 2021 Yann Labbé):

| Constant | Upstream source |
|---|---|
| `CRAVES_JOINT_KP_NAMES`, `CRAVES_VERTEX_SEQ`, `CRAVES_ACTOR_NAME` | `robopose/third_party/craves/get_2d_gt.py` |
| `CRAVES_KEYPOINT_OFFSETS` | `owi-description/keypoints.json` in the [RoboPose deps archive](https://www.paris.inria.fr/archive_ylabbeprojectsdata/robopose/deps/owi-description/keypoints.json) |

They are inlined rather than downloaded because they are a fixed 17-row
definition of what PCK-17 measures. The metric is not reproducible without
them, and they never change. Each is annotated in-source with its origin and,
for the offsets, the upstream SHA-256 so it can be re-verified.

Note this does **not** extend to the OWI-535 meshes: those are the CRAVES CAD
model and are not redistributed by ROBOP or robot-renderer. `make assets`
downloads them.

## Libraries

| Component | How it is used | License |
|---|---|---|
| PyTorch3D | conda package (see `Makefile`) | BSD-3-Clause |
| faiss, kornia, scikit-learn, pyrender, trimesh, etc. | pip dependencies (see `requirements.txt`) | permissive (MIT/BSD/Apache) |

## Datasets

The evaluation datasets (DREAM panda-orb, CtRNet Baxter, CRAVES-lab, Hydra
ICP benchmark) are not distributed with this repository and are subject to
their authors' terms; the eval configs take dataset paths as input.
