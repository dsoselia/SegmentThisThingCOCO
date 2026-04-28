#!/bin/bash
# Usage:
#   sbatch cluster/sam_h100_2gpu_coco_seg_low_res.sbatch

#SBATCH --job-name=sam-h100-2gpu-coco-seg-low-res
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu-h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256000
#SBATCH --time=48:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentAnything/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentAnything
cd "$ROOT"
mkdir -p "$ROOT/slurm"

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

CONFIG="${CONFIG:-configs/sam_vit_b_coco_seg_2xh100_low_res.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/run_sam.py train-sam --config "$CONFIG"
