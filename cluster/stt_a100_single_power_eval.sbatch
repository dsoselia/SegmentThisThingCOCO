#!/bin/bash
# Usage:
#   sbatch cluster/stt_a100_single_power_eval.sbatch

#SBATCH --job-name=stt-a100-power-eval
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu-a100
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
cd "$ROOT"
mkdir -p /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/power_eval

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda initialization script" >&2
  exit 1
fi
conda activate py12

CONFIG="${CONFIG:-configs/stt_b_coco_log_rect_seg_3xa100.json}"
CHECKPOINT="${CHECKPOINT:-/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/runs/stt_b_coco_log_rect_seg_3xa100/seg_20260414T131450Z/checkpoints/checkpoint_step_0018500.pt}"
MANIFEST="${MANIFEST:-/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/data/coco2017/manifests/coco2017_val_instances.jsonl}"
OUTPUT="${OUTPUT:-/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/power_eval/log_rect_stt_b_coco_val100_step18500_a100.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
WARMUP_EXAMPLES="${WARMUP_EXAMPLES:-10}"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "manifest=$MANIFEST"
echo "output=$OUTPUT"
echo "max_examples=$MAX_EXAMPLES"
echo "warmup_examples=$WARMUP_EXAMPLES"

python scripts/measure_inference_power.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --device cuda \
  --max-examples "$MAX_EXAMPLES" \
  --warmup-examples "$WARMUP_EXAMPLES"
