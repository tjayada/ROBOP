#!/usr/bin/env bash
# NeMO perturbation sweep over all six datasets. Magnitudes are defined in
# configs/experiment/ablation_joint_noise.yaml and include the paired zero baseline.
# Usage: NEMO_CKPT=/path/to/checkpoint.pth DATA_ROOT=/data scripts/run_joint_noise_sweep.sh
# Uses detector-free bootstrap crops unless NEMO_DETECTIONS points to a detection directory.
#SBATCH --job-name=joint-noise-sweep
#SBATCH --output=logs/joint_noise_sweep_%j.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HYDRA_FULL_ERROR=1

SEED="${SEED:-0}"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/joint_noise_sweep_$(date +%Y-%m-%d)}"
PANDA_FRAMES="${PANDA_FRAMES:-250}"
NEMO_DETECTIONS="${NEMO_DETECTIONS:-}"   # empty = detector-free
: "${NEMO_CKPT:?set NEMO_CKPT=/path/to/nemo/checkpoint.pth}"

# Per-dataset roots - explicit overrides win, else fall back to $DATA_ROOT/<layout>.
DATA_ROOT="${DATA_ROOT:-}"
PANDA_DATA="${PANDA_DATA:-${DATA_ROOT:+$DATA_ROOT/panda-orb}}"
BAXTER_DATA="${BAXTER_DATA:-${DATA_ROOT:+$DATA_ROOT/baxter-real-dataset}}"
CRAVES_DATA="${CRAVES_DATA:-${DATA_ROOT:+$DATA_ROOT/test_20181024}}"
HYDRA_LBR_DATA="${HYDRA_LBR_DATA:-${DATA_ROOT:+$DATA_ROOT/hydra_eval/lbr}}"
HYDRA_XARM_DATA="${HYDRA_XARM_DATA:-${DATA_ROOT:+$DATA_ROOT/hydra_eval/xarm}}"
HYDRA_MECA_DATA="${HYDRA_MECA_DATA:-${DATA_ROOT:+$DATA_ROOT/hydra_eval/meca}}"

# SMOKE=1 -> small preflight run before the overnight sweep. Forces its
# three values regardless of any other setting, so a smoke run can never write into
# the real sweep dir (own timestamped dir) or run the full grid.
if [ "${SMOKE:-0}" = 1 ]; then
    DATASETS_FILTER="panda_orb"
    PANDA_FRAMES=20
    SWEEP_ROOT="outputs/a1_smoke_$(date +%Y-%m-%d_%H-%M-%S)"
    echo "[SMOKE] forcing DATASETS_FILTER='$DATASETS_FILTER' PANDA_FRAMES=$PANDA_FRAMES SWEEP_ROOT=$SWEEP_ROOT"
fi

# name | eval script | config-name (- = script default) | data folder | detections JSON
DATASETS=(
    "panda_orb   run_eval_panda_orb.py - $PANDA_DATA       panda_orb.json"
    "baxter      run_eval_baxter.py - $BAXTER_DATA      baxter.json"
    "craves      run_eval_craves.py - $CRAVES_DATA      craves.json"
    "hydra_lbr   run_eval_hydra.py      eval_hydra_lbr   $HYDRA_LBR_DATA   lbr_med7.json"
    "hydra_xarm  run_eval_hydra.py      eval_hydra_xarm  $HYDRA_XARM_DATA  xarm.json"
    "hydra_meca  run_eval_hydra.py      eval_hydra_meca  $HYDRA_MECA_DATA  meca.json"
)

echo "Joint-noise sweep -> $SWEEP_ROOT  (seed=$SEED, detections=${NEMO_DETECTIONS:-null/detector-free})"
for entry in "${DATASETS[@]}"; do
    read -r name script cfg data det_json <<< "$entry"
    if [ -n "${DATASETS_FILTER:-}" ] && [[ " $DATASETS_FILTER " != *" $name "* ]]; then
        continue
    fi
    if [ -z "$data" ] || [ ! -d "$data" ]; then
        echo "=== $name: data folder missing or unset ($data) - skipping"
        continue
    fi

    # Mirror run_all_evals.sh's NeMO invocation, then add the perturbation overrides.
    args=(--multirun +experiment=ablation_joint_noise
          seed="$SEED" estimator=nemo checkpoint="$NEMO_CKPT"
          dataset.data_folder="$data"
          hydra.sweep.dir="$SWEEP_ROOT/$name")
    [ "$cfg" != "-" ] && args=(--config-name "$cfg" "${args[@]}")
    [ "$name" = "panda_orb" ] && args+=(num_eval_frames="$PANDA_FRAMES")
    if [ -n "$NEMO_DETECTIONS" ]; then
        det="$NEMO_DETECTIONS/$det_json"
        if [ ! -f "$det" ]; then
            echo "=== $name: detections regime requested but $det missing - skipping"
            continue
        fi
        args+=(detections_path="$det")
    else
        # Detector-free, no CNOS needed. crop_mode is named explicitly because
        # the shipped default is the published bbox regime, which requires a
        # detection per frame.
        args+=(detections_path=null estimator.crop_mode=bootstrap)
    fi

    echo "=== Joint-noise sweep: $name -> $SWEEP_ROOT/$name"
    "$PYTHON" "evaluation/$script" "${args[@]}"
done

echo
echo "Joint-noise sweep complete -> $SWEEP_ROOT"
echo "Analyze (offline, no GPU):"
echo "  $PYTHON analysis/joint_reuse/analyze_perturbation.py $SWEEP_ROOT --plots $SWEEP_ROOT/figs --save-intermediate"
