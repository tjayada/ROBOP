#!/usr/bin/env bash
# NeMO (published regime: crops to an external detection, so detections are
# required; add estimator.crop_mode=bootstrap to run detector-free instead)
# Usage: DATA_ROOT=/data scripts/eval_nemo.sh
#SBATCH --job-name=nemo
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
set -u
PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$0")/..}
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}
# Model caches (torch.hub DINOv2, SAM/FastSAM weights). Override on clusters
# where jobs land on different nodes, e.g. XDG_CACHE_HOME=/home_local/$USER/.cache
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your dataset root}
CHECKPOINT=${CHECKPOINT:-external/NeMO/checkpoints/checkpoint.pth}   # make assemble-checkpoint
DETECTIONS=${DETECTIONS_DIR:-detections/sam}   # crop_mode=bbox needs a detection per frame

# panda_orb
"$PYTHON" evaluation/run_eval_panda_orb.py \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/panda-orb" \
    detections_path="$DETECTIONS/panda_orb.json.gz" \
    checkpoint="$CHECKPOINT"

# baxter
"$PYTHON" evaluation/run_eval_baxter.py \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/baxter-real-dataset" \
    detections_path="$DETECTIONS/baxter.json.gz" \
    checkpoint="$CHECKPOINT"

# craves
"$PYTHON" evaluation/run_eval_craves.py \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/test_20181024" \
    detections_path="$DETECTIONS/craves.json.gz" \
    checkpoint="$CHECKPOINT"

# hydra_lbr
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_lbr \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/hydra_eval/lbr" \
    detections_path="$DETECTIONS/lbr_med7.json.gz" \
    checkpoint="$CHECKPOINT"

# hydra_xarm
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_xarm \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/hydra_eval/xarm" \
    detections_path="$DETECTIONS/xarm.json.gz" \
    checkpoint="$CHECKPOINT"

# hydra_meca
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_meca \
    seed=0 \
    estimator=nemo \
    dataset.data_folder="$DATA_ROOT/hydra_eval/meca" \
    detections_path="$DETECTIONS/meca.json.gz" \
    checkpoint="$CHECKPOINT"
