#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dsoselia/SegmentThisThingLogRect2GB10}"
REPO="${REPO:-$ROOT/segment_this_thing}"
RAW="${COCO_RAW:-/home/dsoselia/CocoDownloader/coco2017/raw}"
MANIFESTS="${MANIFESTS:-$ROOT/data/coco2017/manifests}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
ENV_NAME="${ENV_NAME:-stt-logrect2-gb10}"

mkdir -p "$MANIFESTS" "$LOGDIR"

source /home/dsoselia/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

for required in \
  "$RAW/train2017" \
  "$RAW/val2017" \
  "$RAW/annotations/instances_train2017.json" \
  "$RAW/annotations/instances_val2017.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required COCO path: $required" >&2
    exit 1
  fi
done

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

python - <<PY > "$LOGDIR/coco2017_counts.json"
import json
from pathlib import Path

raw = Path("$RAW")
manifest_dir = Path("$MANIFESTS")
summary = {
    "train_images": len(list((raw / "train2017").glob("*.jpg"))),
    "val_images": len(list((raw / "val2017").glob("*.jpg"))),
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
print(json.dumps(summary, indent=2, sort_keys=True))
PY

cat "$LOGDIR/coco2017_counts.json"
echo "COCO manifest generation complete under $MANIFESTS"
