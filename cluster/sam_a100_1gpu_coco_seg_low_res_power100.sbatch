#!/bin/bash
# Usage:
#   CHECKPOINT=/path/to/checkpoint.pt sbatch cluster/sam_a100_1gpu_coco_seg_low_res_power100.sbatch

#SBATCH --job-name=sam-a100-1gpu-seg-low-res-power100
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128000
#SBATCH --time=01:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentAnything/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentAnything
cd "$ROOT"
mkdir -p "$ROOT/slurm" "$ROOT/artifacts/power"

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/sam_vit_b_coco_seg_1xa100_low_res_power100.json}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/sam_vit_b_coco_seg_2xh100_low_res/seg_20260421T074257Z/checkpoints/checkpoint_step_0017500.pt}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/power/sam_vit_b_coco_seg_1xa100_low_res_power100.json}"
GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 1
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

python scripts/eval_seg_power.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --max-examples 100 \
  --log-every 10 \
  --power-interval-ms 20 \
  --warmup-examples 5
