#!/bin/bash
# Run the selected estimators across the selected datasets.
# Usage: DATA_ROOT=/data ./run_all_evals.sh [model ...]
# Models: nemo | foundpose | gigapose | gigapose_megapose | megapose | megapose_coarse
# Set NUM_EVAL_FRAMES for a smoke subset and SAVE_RESULTS_JSON=0 to omit JSON output.
# Expected DATA_ROOT layout:
#   baxter-real-dataset/   panda-orb/   test_20181024/   hydra_eval/{lbr,xarm,meca}/
#SBATCH --job-name=robop-eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(dirname "$0")}
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}
# Model caches (torch.hub DINOv2, SAM/FastSAM weights). Override on clusters
# where jobs land on different nodes, e.g. XDG_CACHE_HOME=/home_local/$USER/.cache
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}
SEED=${SEED:-0}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your dataset root}

# Subset knob: when set, evaluate NUM_EVAL_FRAMES evenly-spaced frames per
# dataset (per measurement for Hydra). Unset = full frame counts (final numbers).
NUM_EVAL_FRAMES=${NUM_EVAL_FRAMES:-}

# Write per-frame results JSON (poses, runtime, VRAM): 1 = save (default),
# 0 = skip. Independent of NUM_EVAL_FRAMES, so a subset run can still be saved
# for the result tables (a subset already records valid runtime and VRAM).
SAVE_RESULTS_JSON=${SAVE_RESULTS_JSON:-1}

BAXTER_DATA=${BAXTER_DATA:-$DATA_ROOT/baxter-real-dataset}
PANDA_DATA=${PANDA_DATA:-$DATA_ROOT/panda-orb}
CRAVES_DATA=${CRAVES_DATA:-$DATA_ROOT/test_20181024}
HYDRA_LBR_DATA=${HYDRA_LBR_DATA:-$DATA_ROOT/hydra_eval/lbr}
HYDRA_XARM_DATA=${HYDRA_XARM_DATA:-$DATA_ROOT/hydra_eval/xarm}
HYDRA_MECA_DATA=${HYDRA_MECA_DATA:-$DATA_ROOT/hydra_eval/meca}

# Detection JSONs. Set DETECTIONS_DIR to whichever directory holds them; the
# detector writes a sam/ and a fastsam/ set, and ROBOP results use sam.
# Models with needs_detections=1 pass
# detections_path=$DETECTIONS_DIR/<detections-json column of the DATASETS row>.
DETECTIONS_DIR=${DETECTIONS_DIR:-detections/sam}

# Checkpoints (see Makefile: assemble-checkpoint / download-gigapose / download-megapose).
NEMO_CKPT=${NEMO_CKPT:-external/NeMO/checkpoints/checkpoint.pth}
GIGAPOSE_CKPT=${GIGAPOSE_CKPT:-external/gigapose/pretrained/gigaPose_v1.ckpt}
MEGAPOSE_MODELS=${MEGAPOSE_MODELS:-external/gigapose/pretrained/megapose-models}

export HYDRA_FULL_ERROR=1

# ── Models to evaluate - comment in/out, or override via CLI args ───────────
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(
        nemo
        foundpose
        gigapose
        gigapose_megapose
        megapose
        megapose_coarse
    )
fi

# ── Datasets to evaluate - comment in/out ────────────────────────────────────
# columns: name | eval script | hydra config-name (- = script default) |
#          data folder | detections JSON filename under $DETECTIONS_DIR
DATASETS=(
    "panda_orb   run_eval_panda_orb.py  -                $PANDA_DATA       panda_orb.json.gz"
    "baxter      run_eval_baxter.py     -                $BAXTER_DATA      baxter.json.gz"
    "craves      run_eval_craves.py     -                $CRAVES_DATA      craves.json.gz"
    "hydra_lbr   run_eval_hydra.py      eval_hydra_lbr   $HYDRA_LBR_DATA   lbr_med7.json.gz"
    "hydra_xarm  run_eval_hydra.py      eval_hydra_xarm  $HYDRA_XARM_DATA  xarm.json.gz"
    "hydra_meca  run_eval_hydra.py      eval_hydra_meca  $HYDRA_MECA_DATA  meca.json.gz"
)

# Per-model hydra overrides -> sets `extra` + `needs_detections`. NeMO's
# published regime crops to an external detection (crop_mode=bbox), so it needs
# them too; run NeMO detector-free with estimator.crop_mode=bootstrap.
set_model_args() {
    case "$1" in
        nemo)              extra=(checkpoint="$NEMO_CKPT"); needs_detections=1 ;;
        foundpose)         extra=(checkpoint=null); needs_detections=1 ;;
        gigapose)          extra=(checkpoint="$GIGAPOSE_CKPT"); needs_detections=1 ;;
        gigapose_megapose) extra=(checkpoint="$GIGAPOSE_CKPT"
                                  estimator.megapose_models_root="$MEGAPOSE_MODELS")
                           needs_detections=1 ;;
        megapose)          extra=(estimator.megapose_models_root="$MEGAPOSE_MODELS")
                           needs_detections=1 ;;
        megapose_coarse)   extra=(estimator.megapose_models_root="$MEGAPOSE_MODELS")
                           needs_detections=1 ;;
        megapose_refiner)  echo "=== $1: use ./run_megapose_refiner.sh (needs per-dataset init results)"
                           return 1 ;;
        *) echo "Unknown model '$1' - see the MODELS list in this script"; return 1 ;;
    esac
}

for model in "${MODELS[@]}"; do
    set_model_args "$model" || continue
    for entry in "${DATASETS[@]}"; do
        read -r name script cfg data det_json <<< "$entry"
        if [ ! -d "$data" ]; then
            echo "=== $model / $name: data folder missing ($data) - skipping"
            continue
        fi
        args=(seed="$SEED" estimator="$model" dataset.data_folder="$data")
        [ "$cfg" != "-" ] && args=(--config-name "$cfg" "${args[@]}")
        if [ -n "$NUM_EVAL_FRAMES" ]; then
            args+=(num_eval_frames="$NUM_EVAL_FRAMES")
        fi
        if [ "$SAVE_RESULTS_JSON" = 0 ]; then
            args+=(save_results=null)
        fi
        if [ "$needs_detections" = 1 ]; then
            det="$DETECTIONS_DIR/$det_json"
            if [ ! -f "$det" ]; then
                echo "=== $model / $name: detections missing ($det) - skipping ($model needs them)"
                continue
            fi
            args+=(detections_path="$det")
        fi
        echo "=== $model / $name${NUM_EVAL_FRAMES:+ (subset: $NUM_EVAL_FRAMES frames)}"
        "$PYTHON" "evaluation/$script" "${args[@]}" "${extra[@]}"
    done
done

echo "All selected evals finished."
