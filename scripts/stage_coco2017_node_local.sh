#!/bin/bash

set -euo pipefail

PROJECT="${PROJECT:-/beacon-projects/foveatedseg}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/foveatedseg-coco-${USER}}"
LOCK_FILE="${STAGE_ROOT}.lock"
MARKER="$STAGE_ROOT/.complete"
SOURCE_RAW="$PROJECT/data/coco2017/raw"
MANIFEST_NAMES=(
  coco2017_mae_train_images.jsonl
  coco2017_mae_val_images.jsonl
  coco2017_train_instances.jsonl
  coco2017_val_instances.jsonl
)

exec 9>"$LOCK_FILE"
flock 9

if [[ -f "$MARKER" ]]; then
  mkdir -p "$STAGE_ROOT/manifests"
  for name in "${MANIFEST_NAMES[@]}"; do
    if [[ ! -f "$STAGE_ROOT/manifests/$name" ]]; then
      sed "s|$SOURCE_RAW|$STAGE_ROOT/raw|g" \
        "$PROJECT/data/coco2017/manifests/$name" > "$STAGE_ROOT/manifests/$name"
    fi
  done
  echo "COCO stage already complete: $STAGE_ROOT"
  echo "$STAGE_ROOT"
  exit 0
fi

BUILD_ROOT="${STAGE_ROOT}.building-${SLURM_JOB_ID:-$$}"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT/raw" "$BUILD_ROOT/manifests"

echo "Staging COCO to node-local storage on $(hostname)"
echo "source=$PROJECT/downloads/coco2017 destination=$STAGE_ROOT"
START=$(date +%s)
unzip -q "$PROJECT/downloads/coco2017/train2017.zip" -d "$BUILD_ROOT/raw"
unzip -q "$PROJECT/downloads/coco2017/val2017.zip" -d "$BUILD_ROOT/raw"

for name in "${MANIFEST_NAMES[@]}"; do
  sed "s|$SOURCE_RAW|$STAGE_ROOT/raw|g" \
    "$PROJECT/data/coco2017/manifests/$name" > "$BUILD_ROOT/manifests/$name"
done

TRAIN_COUNT=$(find "$BUILD_ROOT/raw/train2017" -maxdepth 1 -type f -name '*.jpg' | wc -l)
VAL_COUNT=$(find "$BUILD_ROOT/raw/val2017" -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$TRAIN_COUNT" -ne 118287 || "$VAL_COUNT" -ne 5000 ]]; then
  echo "Unexpected staged image counts: train=$TRAIN_COUNT val=$VAL_COUNT" >&2
  exit 1
fi

touch "$BUILD_ROOT/.complete"
if [[ -e "$STAGE_ROOT" ]]; then
  echo "Incomplete stage exists at $STAGE_ROOT; refusing to overwrite it" >&2
  exit 1
fi
mv "$BUILD_ROOT" "$STAGE_ROOT"
ELAPSED=$(( $(date +%s) - START ))
echo "COCO staging complete: root=$STAGE_ROOT elapsed_s=$ELAPSED train=$TRAIN_COUNT val=$VAL_COUNT"
df -h "$STAGE_ROOT" | tail -1
echo "$STAGE_ROOT"
