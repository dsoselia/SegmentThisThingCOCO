#!/bin/bash
# Usage:
#   sbatch cluster/sam_a100_3gpu_coco_seg.sbatch

#SBATCH --job-name=sam-a100-3gpu-coco-seg
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=92000
#SBATCH --time=08:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentAnything/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentAnything
cd "$ROOT"
mkdir -p "$ROOT/slurm"

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

CONFIG="${CONFIG:-configs/sam_vit_b_coco_seg_3xa100.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/run_sam.py train-sam --config "$CONFIG"
