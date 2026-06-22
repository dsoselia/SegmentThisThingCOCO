#!/bin/bash
#SBATCH --job-name=stt-lambda-resume-smoke
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_frozen_10k.json"
SOURCE="$PROJECT/runs/logrect2_learnable_lambda/frozen_c10_10k/mae_20260622T082412Z/checkpoints/checkpoint_milestone_0002500.pt"
OUTPUT_ROOT="$PROJECT/runs/logrect2_learnable_lambda/resume_smoke_from_2500"
CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"
mkdir -p "$PROJECT/slurm" "$OUTPUT_ROOT"
cd "$REPO"
source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline

python scripts/build_lambda_continuation_config.py \
  --base-config "$BASE" --output "$CONFIG" --fork-from "$SOURCE" \
  --output-dir "$OUTPUT_ROOT" --profile-name beacon_lambda_resume_2500_smoke \
  --wandb-run-name logrect2-lambda-learnable-from-2500-smoke \
  --unfreeze-step 2500 --num-steps 2505 --save-every 5 --log-every 1 \
  --val-every 5 --val-max-examples 16 --milestone-step 2505

GPU_WATCH="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu_watch.csv"
( while true; do nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$GPU_WATCH"; sleep 5; done ) &
GPU_WATCH_PID=$!
trap 'kill "$GPU_WATCH_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID host=$(hostname) source=$SOURCE config=$CONFIG"
python scripts/run_stt.py pretrain-mae --config "$CONFIG"
LATEST_RUN=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'mae_*' | sort | tail -1)
CHECKPOINT="$LATEST_RUN/checkpoints/checkpoint_milestone_0002505.pt"
python scripts/validate_mae_checkpoint.py --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --expected-completed-steps 2505 --expect-lambda-changed-from 1.0 --minimum-change 1e-7 --expect-learnable
python - "$LATEST_RUN/metrics.jsonl" <<'PY'
import json, sys
rows=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
if len(rows) < 5: raise SystemExit("expected five logged steps")
if not all(row["lambda_learnable"] == 1.0 for row in rows): raise SystemExit("lambda was not learnable")
if not any(row["lambda_grad_abs"] > 0 for row in rows): raise SystemExit("lambda gradient never became nonzero")
if abs(rows[-1]["lambda_scale"] - 1.0) < 1e-7: raise SystemExit("lambda did not move")
print(json.dumps({"initial_lambda": rows[0]["lambda_scale"], "final_lambda": rows[-1]["lambda_scale"], "final_grad_abs": rows[-1]["lambda_grad_abs"]}, indent=2))
PY
find "$LATEST_RUN/wandb" -type f -name 'run-*.wandb' -print -quit | grep -q .
echo "LEARNABLE_RESUME_SMOKE_OK run_dir=$LATEST_RUN"
