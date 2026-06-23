#!/bin/bash
#SBATCH --job-name=stt-seg-lambda-frozen10k
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_seg_from_frozen10k.json"
CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"
OUTPUT_ROOT="$PROJECT/runs/logrect2_learnable_lambda/seg_unlocked_from_frozen_mae10k"
STAGE_COCO="${STAGE_COCO:-1}"
mkdir -p "$PROJECT/slurm" "$OUTPUT_ROOT"
cd "$REPO"
source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

python - "$BASE" "$CONFIG" "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
base, output, root = sys.argv[1:]
config = json.load(open(base))
runs = sorted(Path(root).glob("seg_*"))
if runs and (runs[-1] / "last_checkpoint.txt").exists():
    checkpoint = Path((runs[-1] / "last_checkpoint.txt").read_text().strip())
    if checkpoint.exists(): config["runtime"]["resume_from"] = str(checkpoint)
with open(output, "w") as handle: json.dump(config, handle, indent=2)
PY

if [[ "$STAGE_COCO" == "1" ]]; then
  STAGE_ROOT="/tmp/foveatedseg-coco-${USER}"
  PROJECT="$PROJECT" STAGE_ROOT="$STAGE_ROOT" bash scripts/stage_coco2017_node_local.sh
  python - "$CONFIG" "$STAGE_ROOT" <<'PY'
import json, sys
path, root = sys.argv[1:]
config = json.load(open(path))
config["segmentation"]["train_manifest"] = f"{root}/manifests/coco2017_train_instances.jsonl"
config["evaluation"]["named_eval_manifests"]["coco2017_val"] = f"{root}/manifests/coco2017_val_instances.jsonl"
config["runtime"]["wandb_tags"] = sorted(set(config["runtime"].get("wandb_tags", []) + ["node-local-coco"]))
with open(path, "w") as handle: json.dump(config, handle, indent=2)
PY
fi

GPU_WATCH="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
( while true; do sleep 900; wandb sync "$OUTPUT_ROOT"/seg_*/wandb/offline-run-* >/dev/null 2>&1 || true; done ) &
WANDB_SYNC_PID=$!
trap 'kill "$GPU_WATCH_PID" "$WANDB_SYNC_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG stage_coco=$STAGE_COCO"
set +e
python scripts/run_stt.py train-stt --config "$CONFIG"
STATUS=$?
set -e
wandb sync "$OUTPUT_ROOT"/seg_*/wandb/offline-run-* || true
if [[ "$STATUS" -ne 0 ]]; then exit "$STATUS"; fi
LATEST_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_model.pt" ]]; then
  echo "TRAINING_COMPLETE run_dir=$LATEST_RUN"
else
  echo "Graceful interruption; requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
