# STT Replication Pipeline

This repository now includes a best-effort replication stack under `stt_pipeline/` and `scripts/run_stt.py`.

## What It Adds

- Foveated MAE pretraining for the STT image encoder.
- Segmentation fine-tuning with continuous foveated mask targets.
- Manifest-driven training and evaluation datasets.
- Zero-shot single-point evaluation over manifest-defined datasets.
- Synthetic benchmarking for cluster fit checks.

## Manifest Format

Training and eval both use JSONL manifests. Each line is one image:

```json
{
  "image_path": "/abs/path/to/image.png",
  "dataset_name": "sa1b",
  "segments": [
    { "mask_path": "/abs/path/to/mask_0.png" }
  ]
}
```

Supported segment encodings:

- `mask_path`
- `polygon`
- `rle` if `pycocotools` is installed

## Entry Points

```bash
python scripts/run_stt.py pretrain-mae --config configs/stt_b_smoke.json
python scripts/run_stt.py train-stt --config configs/stt_b_smoke.json
python scripts/run_stt.py eval-stt --config configs/stt_b_smoke.json --checkpoint /path/to/final_model.pt
python scripts/run_stt.py benchmark-stt --config configs/stt_b_smoke.json
```

## Important Assumptions

- The public repo does not provide raw SA-1B ingestion code, so training is manifest-driven.
- Prompt noise is disabled by default because the paper does not publish the noise scale.
- If `cv2` or `scipy` is unavailable, eval point selection falls back to a coarse positive-pixel heuristic.
- If `pycocotools` is unavailable, RLE masks are not supported.

## Actual Run Guidance

- Validate on `STT-B` first.
- Use the smoke config only for correctness checks.
- For real runs, create a separate config with real manifests, longer schedules, and a larger effective batch.
