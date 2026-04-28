#!/bin/bash
# Usage:
#   sbatch cluster/sam_h100_2gpu_coco_mae_low_res.sbatch

#SBATCH --job-name=sam-h100-2gpu-coco-mae-low-res
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu-h100
#SBATCH --nodes=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256000
#SBATCH --time=08:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentAnything/slurm/%x-%j.out

set -euo pipefail

ROOT=/home/dsoselia/scratch.varshney-prj/SegmentAnything
cd "$ROOT"
mkdir -p "$ROOT/slurm" "$ROOT/artifacts"

CONDA_SH="/home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh"
source "$CONDA_SH"
conda activate py12

CONFIG="${CONFIG:-configs/sam_vit_b_coco_mae_2xh100_low_res.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
GPU_WATCH="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
ARTIFACT="${ARTIFACT:-$ROOT/artifacts/sam_vit_b_coco_mae_2xh100_low_res_final_backbone.pt}"

(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/run_sam.py pretrain-mae --config "$CONFIG"

RUN_OUTPUT_DIR="$(python - <<'PY'
import json, os
cfg=os.environ["CONFIG"]
print(json.load(open(cfg))["runtime"]["output_dir"])
PY
)"
LATEST_RUN="$(find "$RUN_OUTPUT_DIR" -maxdepth 1 -type d -name 'mae_*' | sort | tail -n 1)"
cp "$LATEST_RUN/final_backbone.pt" "$ARTIFACT"
