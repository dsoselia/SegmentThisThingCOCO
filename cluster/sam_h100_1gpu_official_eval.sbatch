#!/bin/bash
# Usage:
#   CHECKPOINT=/path/to/sam_vit_b_01ec64.pth sbatch cluster/sam_h100_1gpu_official_eval.sbatch

#SBATCH --job-name=sam-h100-1gpu-official-eval
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu-h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128000
#SBATCH --time=04:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentAnything/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentAnything
cd "$ROOT"
mkdir -p "$ROOT/slurm"

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

CONFIG="${CONFIG:-configs/sam_vit_b_official_eval_coco_val.json}"
CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/sam_vit_b_01ec64.pth}"
GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

python scripts/run_sam.py eval-sam --config "$CONFIG" --checkpoint "$CHECKPOINT"
