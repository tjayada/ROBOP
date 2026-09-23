#!/usr/bin/env bash
# MegaPose coarse only - the SO(3)-grid coarse stage as a coarse model, no refiner
# (needs robot-detector detections; no checkpoint override)
# Usage: DATA_ROOT=/data scripts/eval_megapose_coarse.sh
#SBATCH --job-name=megapose_coarse
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
MEGAPOSE_MODELS=${MEGAPOSE_MODELS:-external/gigapose/pretrained/megapose-models}   # make download-megapose
DETECTIONS=${DETECTIONS_DIR:-detections/sam}

# panda_orb
"$PYTHON" evaluation/run_eval_panda_orb.py \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/panda-orb" \
    detections_path="$DETECTIONS/panda_orb.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"

# baxter
"$PYTHON" evaluation/run_eval_baxter.py \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/baxter-real-dataset" \
    detections_path="$DETECTIONS/baxter.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"

# craves
"$PYTHON" evaluation/run_eval_craves.py \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/test_20181024" \
    detections_path="$DETECTIONS/craves.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"

# hydra_lbr
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_lbr \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/hydra_eval/lbr" \
    detections_path="$DETECTIONS/lbr_med7.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"

# hydra_xarm
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_xarm \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/hydra_eval/xarm" \
    detections_path="$DETECTIONS/xarm.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"

# hydra_meca
"$PYTHON" evaluation/run_eval_hydra.py \
    --config-name eval_hydra_meca \
    seed=0 \
    estimator=megapose_coarse \
    dataset.data_folder="$DATA_ROOT/hydra_eval/meca" \
    detections_path="$DETECTIONS/meca.json.gz" \
    estimator.megapose_models_root="$MEGAPOSE_MODELS"
