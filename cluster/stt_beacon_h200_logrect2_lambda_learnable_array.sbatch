#!/bin/bash
#SBATCH --job-name=stt-lambda-learn
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2%3
#SBATCH --signal=USR1@300
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%A_%a.out
#SBATCH --open-mode=append

set -euo pipefail
PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_frozen_10k.json"
FROZEN_RUN="$PROJECT/runs/logrect2_learnable_lambda/frozen_c10_10k/mae_20260622T082412Z"
STEPS=(2500 5000 7500)
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}
SOURCE="$FROZEN_RUN/checkpoints/checkpoint_milestone_$(printf '%07d' "$STEP").pt"
OUTPUT_ROOT="$PROJECT/runs/logrect2_learnable_lambda/learnable_from_${STEP}"
ARTIFACT_ROOT="$PROJECT/artifacts/logrect2_learnable_lambda"
CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
mkdir -p "$PROJECT/slurm" "$OUTPUT_ROOT" "$ARTIFACT_ROOT"
cd "$REPO"
source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline
STAGE_COCO="${STAGE_COCO:-0}"

SOURCE_ARGS=(--fork-from "$SOURCE")
LATEST_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1 || true)
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/final_encoder.pt" ]]; then echo "already complete: $LATEST_RUN"; exit 0; fi
if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/last_checkpoint.txt" ]]; then
  RESUME=$(tr -d '[:space:]' < "$LATEST_RUN/last_checkpoint.txt")
  if [[ -f "$RESUME" ]]; then SOURCE_ARGS=(--resume-from "$RESUME"); fi
fi

python scripts/build_lambda_continuation_config.py \
  --base-config "$BASE" --output "$CONFIG" "${SOURCE_ARGS[@]}" \
  --output-dir "$OUTPUT_ROOT" --profile-name "beacon_lambda_learnable_from_${STEP}" \
  --wandb-run-name "logrect2-lambda-learnable-from-${STEP}" \
  --unfreeze-step "$STEP" --num-steps 10000 --save-every 500 --log-every 20 \
  --val-every 500 --val-max-examples 128 --milestone-step 10000

if [[ "$STAGE_COCO" == "1" ]]; then
  STAGE_ROOT="/tmp/foveatedseg-coco-${USER}"
  PROJECT="$PROJECT" STAGE_ROOT="$STAGE_ROOT" bash scripts/stage_coco2017_node_local.sh
  python - "$CONFIG" "$STAGE_ROOT" <<'PY'
import json, sys
path, stage_root = sys.argv[1:]
with open(path) as handle:
    config = json.load(handle)
config["mae"]["train_manifest"] = f"{stage_root}/manifests/coco2017_mae_train_images.jsonl"
config["mae"]["val_manifest"] = f"{stage_root}/manifests/coco2017_mae_val_images.jsonl"
config["runtime"]["wandb_tags"] = sorted(set(config["runtime"].get("wandb_tags", []) + ["node-local-coco"]))
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
print(f"node_local_coco={stage_root}")
PY
else
  echo "COCO staging disabled; using project-storage manifests"
fi

GPU_WATCH="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 10; done ) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID step=$STEP host=$(hostname) config=$CONFIG stage_coco=$STAGE_COCO"
set +e
python scripts/run_stt.py pretrain-mae --config "$CONFIG"
STATUS=$?
set -e
if [[ "$STATUS" -ne 0 ]]; then exit "$STATUS"; fi
LATEST_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1)
if [[ -f "$LATEST_RUN/final_encoder.pt" ]]; then
  cp "$LATEST_RUN/final_encoder.pt" "$ARTIFACT_ROOT/stt_b_logrect2_lambda_learnable_from_${STEP}_final_encoder.pt"
  test -f "$LATEST_RUN/checkpoints/checkpoint_milestone_0010000.pt"
  echo "TRAINING_COMPLETE step=$STEP run_dir=$LATEST_RUN"
else
  echo "Graceful interruption; requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
