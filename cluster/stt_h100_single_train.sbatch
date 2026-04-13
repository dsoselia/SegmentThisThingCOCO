#!/bin/bash
# Usage:
#   sbatch cluster/stt_h100_single_train.sbatch
# Optional overrides:
#   sbatch --export=ALL,CONFIG=configs/stt_b_h100_single_stable.json,COMMAND=train-stt cluster/stt_h100_single_train.sbatch

#SBATCH --job-name=stt-h100-single
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThing/segment_this_thing
cd "$ROOT"

mkdir -p /home/dsoselia/scratch.varshney-prj/SegmentThisThing/slurm

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

CONFIG="${CONFIG:-configs/stt_b_h100_single_stable.json}"
COMMAND="${COMMAND:-train-stt}"

echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "which python=$(which python)"
python --version
echo "config=$CONFIG"
echo "command=$COMMAND"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

python scripts/run_stt.py "$COMMAND" --config "$CONFIG"
