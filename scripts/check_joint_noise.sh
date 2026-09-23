#!/usr/bin/env bash
# Joint-noise preflight: about 5*N_FRAMES panda-orb inferences.
# Required: DATA_FOLDER and CHECKPOINT. Optional: PYTHON, PROJECT_ROOT, N_FRAMES.
set -euo pipefail

PYTHON="${PYTHON:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
: "${DATA_FOLDER:?set DATA_FOLDER=/path/to/panda-orb}"
: "${CHECKPOINT:?set CHECKPOINT=/path/to/nemo_checkpoint.pth}"
N_FRAMES="${N_FRAMES:-20}"

cd "$PROJECT_ROOT"
OUT="outputs/joint_noise_checks_$(date +%Y-%m-%d_%H-%M-%S)"

run_eval () {
  local name=$1; shift
  # Detector-free NeMO: crop_mode=bootstrap + detections_path=null, matching
  # run_joint_noise_sweep.sh. The shipped nemo.yaml defaults to crop_mode=bbox,
  # which raises on frame 0 without a detection.
  "$PYTHON" evaluation/run_eval_panda_orb.py \
      dataset.data_folder="$DATA_FOLDER" checkpoint="$CHECKPOINT" \
      detections_path=null estimator.crop_mode=bootstrap \
      num_eval_frames="$N_FRAMES" num_overlays="${OVERLAYS:-0}" \
      hydra.run.dir="$OUT/$name" "$@"
}

echo "== Module self-test =="
"$PYTHON" evaluation/joint_noise.py

echo "== Joint-limit check (expect 6x OK) =="
"$PYTHON" evaluation/joint_noise.py --check-qlim

echo "== No-op equivalence ('~joint_noise' vs magnitude_deg=0.0) =="
run_eval noop_absent '~joint_noise'
run_eval noop_zero   joint_noise.magnitude_deg=0.0
"$PYTHON" scripts/compare_joint_noise_runs.py --expect-no-joint-noise \
    "$OUT/noop_absent/results.json" "$OUT/noop_zero/results.json"

echo "== Determinism at 2.0 deg (two identical runs) =="
run_eval det_a joint_noise.magnitude_deg=2.0
run_eval det_b joint_noise.magnitude_deg=2.0
"$PYTHON" scripts/compare_joint_noise_runs.py --expect-joint-noise --expect-magnitude 2.0 \
    "$OUT/det_a/results.json" "$OUT/det_b/results.json"

echo "== 10 deg run with overlays for visual inspection =="
OVERLAYS=3 run_eval eyeball_10deg joint_noise.magnitude_deg=10.0
echo "   -> inspect debug_templates/overlay_*.png: predicted keypoints should sit visibly off the arm"
echo "   -> and confirm a 'joint_noise' block (realized_norm_deg ~10) in $OUT/eyeball_10deg/results.json"

echo
echo "ALL CHECKS COMPLETE - artifacts in $OUT"
