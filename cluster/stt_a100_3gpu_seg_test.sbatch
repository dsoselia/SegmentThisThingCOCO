#!/bin/bash
# Usage:
#   sbatch cluster/stt_a100_3gpu_seg_test.sbatch
# Optional overrides:
#   sbatch --export=ALL,CONFIG=configs/stt_b_subset_paper_seg_test_3xa100.json,NPROC_PER_NODE=3 cluster/stt_a100_3gpu_seg_test.sbatch

#SBATCH --job-name=stt-a100-3gpu-seg-test
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --gpus=a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=92000
#SBATCH --time=03:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
cd "$ROOT"

mkdir -p /home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm

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

CONFIG="${CONFIG:-configs/stt_b_subset_paper_seg_test_3xa100.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
GPU_WATCH="/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "which python=$(which python)"
python --version
echo "config=$CONFIG"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "gpu_watch=$GPU_WATCH"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/run_stt.py train-stt --config "$CONFIG"
