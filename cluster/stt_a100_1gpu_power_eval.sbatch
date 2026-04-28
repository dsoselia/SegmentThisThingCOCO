#!/bin/bash
# Measure single-GPU STT inference-only power on a fixed eval subset.
#
# Example:
#   sbatch --export=ALL,CONFIG=configs/stt_b_coco_paper_seg_3xa100.json,CHECKPOINT=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/runs/stt_b_coco_paper_seg_3xa100/seg_20260413T105833Z/checkpoints/checkpoint_step_0016500.pt cluster/stt_a100_1gpu_power_eval.sbatch

#SBATCH --job-name=stt-a100-power-infer
#SBATCH --partition=gpu-a100
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm/stt-a100-power-infer-%j.out

set -euo pipefail

source /home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh
conda activate py12

cd /home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing

CONFIG="${CONFIG:-configs/stt_b_coco_paper_seg_3xa100.json}"
CHECKPOINT="${CHECKPOINT:-/home/dsoselia/scratch.varshney-prj/SegmentThisThing/runs/stt_b_coco_paper_seg_3xa100/seg_20260413T105833Z/checkpoints/checkpoint_step_0016500.pt}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
SAMPLE_INTERVAL_MS="${SAMPLE_INTERVAL_MS:-20}"
OUT_DIR="${OUT_DIR:-/home/dsoselia/scratch.varshney-prj/SegmentThisThing/runs/power_eval/stt_single_a100_100samples_inference_only}"
OUT_JSON="$OUT_DIR/power_summary_${SLURM_JOB_ID}.json"

mkdir -p "$OUT_DIR"

python scripts/power_inference_stt.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --max-examples "$MAX_EXAMPLES" \
  --sample-interval-ms "$SAMPLE_INTERVAL_MS" \
  --output "$OUT_JSON"

echo "Wrote power summary to $OUT_JSON"
