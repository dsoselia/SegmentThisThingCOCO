#!/bin/bash
#SBATCH --job-name=stt-h100-single-overnight
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu-h100
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/segment_this_thing
cd "$ROOT"

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

echo "job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "which python=$(which python)"
python --version
echo "config=configs/stt_b_h100_single_overnight.json"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

python scripts/run_stt.py train-stt --config configs/stt_b_h100_single_overnight.json
