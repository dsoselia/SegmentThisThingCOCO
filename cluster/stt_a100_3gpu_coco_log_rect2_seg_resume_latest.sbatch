#!/bin/bash
#SBATCH --job-name=stt-a100-3gpu-coco-logrect2-seg-resume
#SBATCH -A varshney-prj-aac
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=a100:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=92000
#SBATCH --time=08:00:00
#SBATCH --output=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect2/slurm/%x-%j.out

set -euo pipefail
ROOT=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect2/segment_this_thing
BASE=/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect2
cd "$ROOT"
mkdir -p "$BASE/slurm"
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
export WANDB_MODE=offline
export WANDB_PROJECT=SegmentThisThingLogRect2Zaratan
export TOKENIZERS_PARALLELISM=false

RUN_ROOT="$BASE/runs/stt_b_coco_log_rect2_seg_3xa100"
LATEST_RUN="$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -n 1)"
if [[ -z "$LATEST_RUN" || ! -f "$LATEST_RUN/last_checkpoint.txt" ]]; then
  echo "Could not find resumable run with last_checkpoint.txt under $RUN_ROOT" >&2
  exit 1
fi
RESUME_FROM="$(cat "$LATEST_RUN/last_checkpoint.txt")"
if [[ ! -f "$RESUME_FROM" ]]; then
  echo "Checkpoint from last_checkpoint.txt does not exist: $RESUME_FROM" >&2
  exit 1
fi
CONFIG="${CONFIG:-$BASE/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume_config.json}"
python - "$RESUME_FROM" "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
resume_from = sys.argv[1]
out = Path(sys.argv[2])
base = Path('/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect2/segment_this_thing/configs/stt_b_coco_log_rect2_seg_3xa100.json')
data = json.loads(base.read_text())
data['runtime']['resume_from'] = resume_from
data['runtime']['wandb_tags'] = sorted(set(data['runtime'].get('wandb_tags', []) + ['resume']))
out.write_text(json.dumps(data, indent=2) + '\n')
print(out)
PY
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
GPU_WATCH="$BASE/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
echo "job_id=$SLURM_JOB_ID"
echo "hostname=$(hostname)"
echo "latest_run=$LATEST_RUN"
echo "resume_from=$RESUME_FROM"
echo "config=$CONFIG"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"
    sleep 5
  done
) &
GPU_WATCH_PID=$!
trap 'kill $GPU_WATCH_PID 2>/dev/null || true' EXIT
torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" scripts/run_stt.py train-stt --config "$CONFIG"
