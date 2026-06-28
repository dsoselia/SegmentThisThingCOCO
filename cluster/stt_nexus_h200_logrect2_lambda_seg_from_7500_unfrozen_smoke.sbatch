#!/bin/bash
# Segmentation smoke - 3 steps + 1 resumed step, 1x H200 SXM, learned lambda from MAE 7500-unfreeze run.

#SBATCH --job-name=stt-seg-lambda75-smoke
#SBATCH --output=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO/slurm/%x-%j.out
#SBATCH --error=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --exclude=cml36
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

ROOT=/cmlscratch/dsoselia/logrect_learnable/SegmentThisThingCOCO
CONFIG="$ROOT/configs/stt_b_nexus_h200_logrect2_lambda_seg_from_7500_unfrozen_smoke.json"
CHECKPOINT="$ROOT/artifacts/logrect2_learnable_lambda/learnable_from_7500/mae_20260622T235831Z/checkpoints/checkpoint_milestone_0010000.pt"
OUTPUT_ROOT="$ROOT/runs/logrect2_learnable_lambda/seg_smoke_from_7500_unfrozen_mae10k"
mkdir -p "$ROOT/slurm" "$OUTPUT_ROOT"

source /cmlscratch/dsoselia/miniconda3/etc/profile.d/conda.sh
conda activate py12

cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: MAE checkpoint not found at $CHECKPOINT"
  exit 1
fi

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python scripts/run_stt.py train-stt --config "$CONFIG"
RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

run = Path(sys.argv[1])
rows = [json.loads(line) for line in open(run / "metrics.jsonl") if line.strip()]
if len(rows) != 3:
    raise SystemExit(f"expected 3 metrics rows, found {len(rows)}")
if [row["step"] for row in rows] != [0, 1, 2]:
    raise SystemExit(f"unexpected steps: {[row['step'] for row in rows]}")
if not all(row["lambda_learnable"] == 1.0 for row in rows):
    raise SystemExit("lambda was not learnable")
if not any(math.isfinite(row["lambda_grad_abs"]) and row["lambda_grad_abs"] > 0 for row in rows):
    raise SystemExit("lambda gradient was never finite and nonzero")
if abs(rows[0]["lambda_scale"] - 0.7517207) > 0.02:
    raise SystemExit(f"lambda did not load from learned checkpoint: {rows[0]['lambda_scale']}")
if abs(rows[-1]["lambda_scale"] - rows[0]["lambda_scale"]) < 1e-8:
    raise SystemExit("lambda did not change")
checkpoint = torch.load(run / "checkpoints/checkpoint_step_0000002.pt", map_location="cpu", weights_only=False)
raw = checkpoint["model"]["foveator.raw_lambda_scale"]
if not torch.isfinite(raw):
    raise SystemExit("checkpoint lambda is non-finite")
if not any(key.startswith("segment_model.") for key in checkpoint["model"]):
    raise SystemExit("joint model state missing")
print(json.dumps({
    "run_dir": str(run),
    "initial_lambda": rows[0]["lambda_scale"],
    "final_lambda": rows[-1]["lambda_scale"],
    "final_grad": rows[-1]["lambda_grad_abs"],
}, indent=2))
PY

RESUME_CONFIG="$ROOT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
python - "$CONFIG" "$RESUME_CONFIG" "$RUN_DIR/checkpoints/checkpoint_step_0000002.pt" <<'PY'
import json
import sys

source, output, checkpoint = sys.argv[1:]
config = json.load(open(source))
config["runtime"]["resume_from"] = checkpoint
config["runtime"]["num_steps"] = 4
with open(output, "w") as handle:
    json.dump(config, handle, indent=2)
PY

python scripts/run_stt.py train-stt --config "$RESUME_CONFIG"
python - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

import torch

run = Path(sys.argv[1])
rows = [json.loads(line) for line in open(run / "metrics.jsonl") if line.strip()]
if [row["step"] for row in rows] != [0, 1, 2, 3]:
    raise SystemExit(f"unexpected resumed steps: {[row['step'] for row in rows]}")
before = torch.load(run / "checkpoints/checkpoint_step_0000002.pt", map_location="cpu", weights_only=False)
after = torch.load(run / "checkpoints/checkpoint_step_0000003.pt", map_location="cpu", weights_only=False)
raw_before = before["model"]["foveator.raw_lambda_scale"].item()
raw_after = after["model"]["foveator.raw_lambda_scale"].item()
if not all(map(math.isfinite, (raw_before, raw_after))):
    raise SystemExit("non-finite resumed lambda")
if raw_before == raw_after:
    raise SystemExit("lambda did not continue changing after resume")
print(json.dumps({"resume_step": 3, "raw_lambda_before": raw_before, "raw_lambda_after": raw_after}, indent=2))
PY

echo "SEGMENTATION_LEARNABLE_LAMBDA_7500_SMOKE_OK run_dir=$RUN_DIR"
