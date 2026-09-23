#!/usr/bin/env bash
# Refine poses from an existing evaluation results JSON.
# Edit the dataset/result pairs at the bottom, then run with DATA_ROOT=/data.
#SBATCH --job-name=robop-megapose-refiner
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

BAXTER_DATA=${BAXTER_DATA:-$DATA_ROOT/baxter-real-dataset}
PANDA_DATA=${PANDA_DATA:-$DATA_ROOT/panda-orb}
CRAVES_DATA=${CRAVES_DATA:-$DATA_ROOT/test_20181024}
HYDRA_LBR_DATA=${HYDRA_LBR_DATA:-$DATA_ROOT/hydra_eval/lbr}
HYDRA_XARM_DATA=${HYDRA_XARM_DATA:-$DATA_ROOT/hydra_eval/xarm}
HYDRA_MECA_DATA=${HYDRA_MECA_DATA:-$DATA_ROOT/hydra_eval/meca}

MEGAPOSE_MODELS=${MEGAPOSE_MODELS:-external/gigapose/pretrained/megapose-models}

export HYDRA_FULL_ERROR=1

# refine <dataset> <init results.json>
refine() {
    local ds=$1 init=$2
    local script cfg data
    case "$ds" in
        panda_orb)  script=run_eval_panda_orb.py; cfg=-;               data=$PANDA_DATA ;;
        baxter)     script=run_eval_baxter.py;    cfg=-;               data=$BAXTER_DATA ;;
        craves)     script=run_eval_craves.py;    cfg=-;               data=$CRAVES_DATA ;;
        hydra_lbr)  script=run_eval_hydra.py;     cfg=eval_hydra_lbr;  data=$HYDRA_LBR_DATA ;;
        hydra_xarm) script=run_eval_hydra.py;     cfg=eval_hydra_xarm; data=$HYDRA_XARM_DATA ;;
        hydra_meca) script=run_eval_hydra.py;     cfg=eval_hydra_meca; data=$HYDRA_MECA_DATA ;;
        *) echo "refine: unknown dataset '$ds'"; return 1 ;;
    esac
    if [ ! -d "$data" ]; then
        echo "=== megapose_refiner / $ds: data folder missing ($data) - skipping"
        return 0
    fi
    if [ ! -f "$init" ]; then
        echo "=== megapose_refiner / $ds: init results missing ($init) - skipping"
        return 0
    fi
    local args=(seed="$SEED" estimator=megapose_refiner
                dataset.data_folder="$data"
                estimator.init_results_path="$init"
                estimator.megapose_models_root="$MEGAPOSE_MODELS"
                checkpoint=null)
    [ "$cfg" != "-" ] && args=(--config-name "$cfg" "${args[@]}")
    echo "=== megapose_refiner / $ds  (init: $init)"
    "$PYTHON" "evaluation/$script" "${args[@]}"
}

# ── Refinement runs: dataset + the results.json to refine ───────────────────
# (init results come from outputs/<run>/results.json of a previous eval run
# on the SAME dataset - comment in and point at your runs)
# refine hydra_xarm  outputs/<coarse-run>/results.json
# refine hydra_lbr   outputs/<run>/results.json
# refine hydra_meca  outputs/<run>/results.json
# refine baxter      outputs/<run>/results.json
# refine craves      outputs/<run>/results.json
# refine panda_orb   outputs/<run>/results.json

echo "All refinement runs finished."
