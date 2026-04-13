#!/bin/bash
# Usage:
#   sbatch cluster/stt_a100_3gpu_coco_mae.sbatch

#SBATCH --job-name=stt-a100-3gpu-coco-mae
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --gpus=a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=92000
#SBATCH --time=08:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing
cd "$ROOT"
mkdir -p /home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm /home/dsoselia/scratch.varshney-prj/SegmentThisThing/artifacts

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

CONFIG="${CONFIG:-configs/stt_b_coco_paper_mae_3xa100.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
GPU_WATCH="/home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
ARTIFACT="/home/dsoselia/scratch.varshney-prj/SegmentThisThing/artifacts/stt_b_coco_paper_mae_3xa100_final_encoder.pt"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "config=$CONFIG"
echo "artifact=$ARTIFACT"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/run_stt.py pretrain-mae --config "$CONFIG"

LATEST_RUN="$(find /home/dsoselia/scratch.varshney-prj/SegmentThisThing/runs/stt_b_coco_paper_mae_3xa100 -maxdepth 1 -type d -name 'mae_*' | sort | tail -n 1)"
cp "$LATEST_RUN/final_encoder.pt" "$ARTIFACT"
