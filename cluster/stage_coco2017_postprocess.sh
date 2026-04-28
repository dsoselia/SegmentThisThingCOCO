#!/usr/bin/env bash
set -euo pipefail

BASE=/home/dsoselia/scratch.varshney-prj/SegmentThisThing
REPO="$BASE/segment_this_thing"
RAW="$BASE/data/coco2017/raw"
MANIFESTS="$BASE/data/coco2017/manifests"
LOGDIR="$BASE/data/coco2017/logs"

mkdir -p "$MANIFESTS" "$LOGDIR"

source /home/dsoselia/scratch.varshney-prj/miniconda3/etc/profile.d/conda.sh
conda activate py12

echo "WAIT_START $(date -Iseconds)"
while [[ ! -d "$RAW/train2017" ]] || [[ ! -d "$RAW/val2017" ]] || [[ ! -f "$RAW/annotations/instances_train2017.json" ]] || [[ ! -f "$RAW/annotations/instances_val2017.json" ]]; do
  sleep 30
done

while [[ $(find "$RAW/train2017" -maxdepth 1 -type f -name '*.jpg' | wc -l) -lt 118287 ]] || [[ $(find "$RAW/val2017" -maxdepth 1 -type f -name '*.jpg' | wc -l) -lt 5000 ]]; do
  sleep 30
done

echo "RAW_READY $(date -Iseconds)"

cd "$REPO"

python scripts/make_image_manifest.py \
  --images-dir "$RAW/train2017" \
  --output "$MANIFESTS/coco2017_mae_train_images.jsonl" \
  --dataset-name coco2017 \
  --split-name train

python scripts/make_image_manifest.py \
  --images-dir "$RAW/val2017" \
  --output "$MANIFESTS/coco2017_mae_val_images.jsonl" \
  --dataset-name coco2017 \
  --split-name val

python scripts/make_instance_manifest.py \
  --annotations "$RAW/annotations/instances_train2017.json" \
  --images-dir "$RAW/train2017" \
  --output "$MANIFESTS/coco2017_train_instances.jsonl" \
  --dataset-name coco2017 \
  --split-name train

python scripts/make_instance_manifest.py \
  --annotations "$RAW/annotations/instances_val2017.json" \
  --images-dir "$RAW/val2017" \
  --output "$MANIFESTS/coco2017_val_instances.jsonl" \
  --dataset-name coco2017 \
  --split-name val

python scripts/validate_stt_manifest.py \
  --manifest "$MANIFESTS/coco2017_mae_train_images.jsonl" \
  --kind mae > "$LOGDIR/validate_coco2017_mae_train.json"

python scripts/validate_stt_manifest.py \
  --manifest "$MANIFESTS/coco2017_train_instances.jsonl" \
  --kind seg > "$LOGDIR/validate_coco2017_seg_train.json"

python scripts/validate_stt_manifest.py \
  --manifest "$MANIFESTS/coco2017_val_instances.jsonl" \
  --kind seg > "$LOGDIR/validate_coco2017_seg_val.json"

python - <<'PY' > "$LOGDIR/coco2017_counts.json"
import json
from pathlib import Path

base = Path("/home/dsoselia/scratch.varshney-prj/SegmentThisThing/data/coco2017")
manifest_dir = base / "manifests"
raw_dir = base / "raw"

summary = {
    "train_images": len(list((raw_dir / "train2017").glob("*.jpg"))),
    "val_images": len(list((raw_dir / "val2017").glob("*.jpg"))),
}
for name in [
    "coco2017_mae_train_images.jsonl",
    "coco2017_mae_val_images.jsonl",
    "coco2017_train_instances.jsonl",
    "coco2017_val_instances.jsonl",
]:
    path = manifest_dir / name
    with path.open() as handle:
        summary[name] = sum(1 for _ in handle)

print(json.dumps(summary, indent=2))
PY

echo "POSTPROCESS_DONE $(date -Iseconds)"
