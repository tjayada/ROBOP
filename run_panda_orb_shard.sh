#!/bin/bash
# Run one Panda-ORB shard. A machine-specific scheduler launcher should invoke
# this script once for every K in 0..M-1 with one shared OUTPUT_ROOT.
#
#   MODEL=nemo K=0 M=8 PANDA_DATA=/data/panda-orb \
#   DETECTIONS_PATH=/data/panda_orb.json.gz OUTPUT_ROOT=outputs/panda_nemo \
#   ./run_panda_orb_shard.sh
#
# Merge after every shard finished:
#   python evaluation/merge_shards.py --shards \
#     outputs/panda_nemo/shard_*/results.json \
#     --output outputs/panda_nemo/merged/results.json
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$SCRIPT_DIR}
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-python}
MODEL=${MODEL:?set MODEL to a supported estimator name}
K=${K:?set K to the zero-based shard index}
M=${M:?set M to the total number of shards}
PANDA_DATA=${PANDA_DATA:?set PANDA_DATA to the panda-orb directory}
DETECTIONS_PATH=${DETECTIONS_PATH:?set DETECTIONS_PATH to panda_orb.json or .json.gz}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to one directory shared by this shard set}
FRAMES=${FRAMES:-null}
SEED=${SEED:-0}

if ! [[ "$K" =~ ^[0-9]+$ && "$M" =~ ^[1-9][0-9]*$ ]] || ((K >= M)); then
    echo "invalid shard K=$K M=$M; require integers with 0 <= K < M" >&2
    exit 1
fi

NEMO_CKPT=${NEMO_CKPT:-external/NeMO/checkpoints/checkpoint.pth}
GIGAPOSE_CKPT=${GIGAPOSE_CKPT:-external/gigapose/pretrained/gigaPose_v1.ckpt}
MEGAPOSE_MODELS=${MEGAPOSE_MODELS:-external/gigapose/pretrained/megapose-models}

case "$MODEL" in
    nemo)              extra=(checkpoint="$NEMO_CKPT") ;;
    foundpose)         extra=(checkpoint=null) ;;
    gigapose)          extra=(checkpoint="$GIGAPOSE_CKPT") ;;
    gigapose_megapose) extra=(checkpoint="$GIGAPOSE_CKPT"
                              estimator.megapose_models_root="$MEGAPOSE_MODELS") ;;
    megapose|megapose_coarse)
                        extra=(estimator.megapose_models_root="$MEGAPOSE_MODELS") ;;
    *) echo "unsupported MODEL=$MODEL" >&2; exit 1 ;;
esac

export HYDRA_FULL_ERROR=1
echo "=== $MODEL | Panda-ORB shard $K/$M | frames=$FRAMES | $OUTPUT_ROOT/shard_$K"
"$PYTHON" evaluation/run_eval_panda_orb.py \
    seed="$SEED" \
    estimator="$MODEL" \
    dataset.data_folder="$PANDA_DATA" \
    detections_path="$DETECTIONS_PATH" \
    num_eval_frames="$FRAMES" \
    frame_shard="${K}/${M}" \
    num_overlays=0 \
    hydra.run.dir="$OUTPUT_ROOT/shard_$K" \
    "${extra[@]}"
