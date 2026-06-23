#!/bin/bash
#SBATCH --job-name=stt-seg-lambda-smoke
#SBATCH --partition=beacon
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/beacon-projects/foveatedseg/slurm/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
PROJECT=/beacon-projects/foveatedseg
REPO="$PROJECT/src/SegmentThisThingCOCO-logrect2_parameter_search"
BASE="$REPO/configs/stt_b_beacon_h200_logrect2_lambda_seg_from_frozen10k_smoke.json"
CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"
OUTPUT_ROOT="$PROJECT/runs/logrect2_learnable_lambda/seg_smoke_from_frozen_mae10k"
STAGE_COCO="${STAGE_COCO:-1}"
mkdir -p "$PROJECT/slurm" "$OUTPUT_ROOT"
cd "$REPO"
source /beacon-homes/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate stt-coco
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled

cp "$BASE" "$CONFIG"
if [[ "$STAGE_COCO" == "1" ]]; then
  STAGE_ROOT="/tmp/foveatedseg-coco-${USER}"
  PROJECT="$PROJECT" STAGE_ROOT="$STAGE_ROOT" bash scripts/stage_coco2017_node_local.sh
  python - "$CONFIG" "$STAGE_ROOT" <<'PY'
import json, sys
path, root = sys.argv[1:]
config = json.load(open(path))
config["segmentation"]["train_manifest"] = f"{root}/manifests/coco2017_train_instances.jsonl"
with open(path, "w") as handle: json.dump(config, handle, indent=2)
PY
fi

echo "job_id=$SLURM_JOB_ID host=$(hostname) config=$CONFIG stage_coco=$STAGE_COCO"
python scripts/run_stt.py train-stt --config "$CONFIG"
RUN_DIR=$(find "$OUTPUT_ROOT" -maxdepth 1 -type d -name 'seg_*' | sort | tail -1)
python - "$RUN_DIR" <<'PY'
import json, math, sys, torch
from pathlib import Path
run = Path(sys.argv[1])
rows = [json.loads(line) for line in open(run / "metrics.jsonl") if line.strip()]
if len(rows) != 3: raise SystemExit(f"expected 3 metrics rows, found {len(rows)}")
if not all(row["lambda_learnable"] == 1.0 for row in rows): raise SystemExit("lambda was not learnable")
if not any(math.isfinite(row["lambda_grad_abs"]) and row["lambda_grad_abs"] > 0 for row in rows):
    raise SystemExit("lambda gradient was never finite and nonzero")
if abs(rows[-1]["lambda_scale"] - 1.0) < 1e-7: raise SystemExit("lambda did not change")
checkpoint = torch.load(run / "checkpoints/checkpoint_step_0000002.pt", map_location="cpu", weights_only=False)
raw = checkpoint["model"]["foveator.raw_lambda_scale"]
if not torch.isfinite(raw): raise SystemExit("checkpoint lambda is non-finite")
if not any(key.startswith("segment_model.") for key in checkpoint["model"]): raise SystemExit("joint model state missing")
print(json.dumps({"run_dir": str(run), "initial_lambda": rows[0]["lambda_scale"], "final_lambda": rows[-1]["lambda_scale"], "final_grad": rows[-1]["lambda_grad_abs"]}, indent=2))
PY
CHECKPOINT="$RUN_DIR/checkpoints/checkpoint_step_0000002.pt"
RESUME_CONFIG="$PROJECT/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-resume.json"
python - "$CONFIG" "$RESUME_CONFIG" "$CHECKPOINT" <<'PY'
import json, sys
source, output, checkpoint = sys.argv[1:]
config = json.load(open(source))
config["runtime"]["resume_from"] = checkpoint
config["runtime"]["num_steps"] = 4
with open(output, "w") as handle: json.dump(config, handle, indent=2)
PY
python scripts/run_stt.py train-stt --config "$RESUME_CONFIG"
python - "$RUN_DIR" <<'PY'
import json, math, sys, torch
from pathlib import Path
run = Path(sys.argv[1])
rows = [json.loads(line) for line in open(run / "metrics.jsonl") if line.strip()]
if [row["step"] for row in rows] != [0, 1, 2, 3]: raise SystemExit(f"unexpected resumed steps: {[row['step'] for row in rows]}")
before = torch.load(run / "checkpoints/checkpoint_step_0000002.pt", map_location="cpu", weights_only=False)
after = torch.load(run / "checkpoints/checkpoint_step_0000003.pt", map_location="cpu", weights_only=False)
raw_before = before["model"]["foveator.raw_lambda_scale"].item()
raw_after = after["model"]["foveator.raw_lambda_scale"].item()
if not all(map(math.isfinite, (raw_before, raw_after))): raise SystemExit("non-finite resumed lambda")
if raw_before == raw_after: raise SystemExit("lambda did not continue changing after resume")
print(json.dumps({"resume_step": 3, "raw_lambda_before": raw_before, "raw_lambda_after": raw_after}, indent=2))
PY
echo "SEGMENTATION_LEARNABLE_LAMBDA_SMOKE_OK run_dir=$RUN_DIR"
