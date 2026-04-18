# Log-Rectilinear Changes

This note summarizes exactly what we changed in the repo to run a log-rectilinear version of Segment This Thing (STT), using the existing COCO-based training setup as the baseline.

## 1. Goal

Replace the original STT ring-based foveation/tokenization path with a log-rectilinear box tokenizer, while keeping the rest of the experiment as close as possible to the existing COCO STT recipe.

The intended comparison is:
- same model family: `STT-B`
- same dataset setup: COCO MAE pretraining + COCO segmentation finetuning
- same cluster shape: `3 x A100`
- same optimizer/schedule/batch settings as the canonical COCO runs
- only tokenizer/foveation geometry changed

## 2. Baseline vs Rectilinear

### Baseline STT in this repo

The original pipeline used:
- `tokenizer_type = "stt_ring"`
- nested ring foveation
- standard STT foveator path

### New rectilinear path

We added a second tokenizer path:
- `tokenizer_type = "log_rect_box"`

This uses:
- axis-aligned log-rectilinear bins instead of the ring layout
- the repo’s existing `LogRectilinearFoveator`
- the same STT model/decoder/training code around it

## 3. Code Changes

### 3.1 Integrated log-rectilinear tokenizer into the training pipeline

File:
- [stt_pipeline/modeling.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/modeling.py)

Main changes:
- imported `LogRectilinearFoveator`
- updated `build_foveator()` to support both:
  - `stt_ring`
  - `log_rect_box`
- updated `build_model()` so it accepts either foveator type and uses `get_num_tokens()`

Effect:
- benchmarking, MAE pretraining, segmentation training, and eval can now build a log-rectilinear tokenizer from config instead of hard-failing on unsupported tokenizer type

### 3.2 Added log-rect tokenizer parameters to the public config path

Relevant config fields were already present in the dataclass and are now used end-to-end:
- `model.tokenizer_type`
- `model.log_rect_axis_bins`
- `model.log_rect_exponent`
- `model.log_rect_center_width`
- optional `model.pattern_size`

Files:
- [stt_pipeline/config.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/config.py)
- [stt_pipeline/modeling.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/modeling.py)

### 3.3 Added default crop-size inference for log-rect

File:
- [stt_pipeline/modeling.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/modeling.py)

We added logic so that if `pattern_size` is omitted, it is inferred from the standard STT receptive field:
- `token_size * last_stride * last_grid_size`

For the default STT-B geometry in this repo:
- `16 * 8 * 10 = 1280`

Reason:
- keep the log-rect experiment on the same effective crop size as the baseline STT COCO runs
- avoid forcing every log-rect config to restate `pattern_size`

### 3.4 Fixed manifest path handling for portable server runs

File:
- [stt_pipeline/data.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/data.py)

We added:
- `resolve_manifest_path()`

And used it when reading:
- `image_path`
- `mask_path`

Effect:
- manifests with relative paths now resolve relative to the manifest location
- smoke configs no longer depend on a local `/Users/...` path
- fresh server-side clones can use the checked-in smoke manifests directly

This was not specific to log-rectilinear math, but it was required to make the new server-side smoke and cluster runs work reliably.

## 4. Log-Rect Geometry Used

For the first experiment we used the existing prototype geometry:
- `axis_bins = 13`
- `exponent = 4.0`
- `center_width = 16`
- inferred `pattern_size = 1280`

Configs:
- [configs/stt_b_log_rect_box_smoke.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_log_rect_box_smoke.json)
- [configs/stt_b_coco_log_rect_mae_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_log_rect_mae_3xa100.json)
- [configs/stt_b_coco_log_rect_seg_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_log_rect_seg_3xa100.json)

Token count:
- baseline ring STT-B: `172`
- log-rect prototype used here: `169`

Interpretation:
- the comparison is intentionally tokenizer-only and near-token-matched

## 5. New Experiment Configs Added

### 5.1 MAE pretraining config

Added:
- [configs/stt_b_coco_log_rect_mae_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_log_rect_mae_3xa100.json)

This mirrors the canonical COCO MAE run except for tokenizer choice.

Kept the same as baseline:
- `STT-B`
- `3 x A100`
- `micro_batch_size = 8`
- `effective_batch_size = 1024`
- same optimizer and schedule
- same COCO MAE train manifest

Changed:
- `tokenizer_type = "log_rect_box"`
- log-rect geometry fields
- output/profile/artifact names moved to log-rect-specific paths

### 5.2 Segmentation config

Added:
- [configs/stt_b_coco_log_rect_seg_3xa100.json](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/configs/stt_b_coco_log_rect_seg_3xa100.json)

This mirrors the canonical COCO segmentation run except for tokenizer choice and encoder initialization path.

Kept the same as baseline:
- `STT-B`
- `3 x A100`
- `micro_batch_size = 2`
- `effective_batch_size = 2048`
- same optimizer/schedule
- same COCO train/val manifests
- same save/eval cadence

Changed:
- `tokenizer_type = "log_rect_box"`
- log-rect geometry fields
- `pretrained_encoder` points to the log-rect MAE artifact, not the released STT checkpoint

## 6. New Cluster Launchers Added

Added:
- [cluster/stt_a100_3gpu_coco_log_rect_mae.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_log_rect_mae.sbatch)
- [cluster/stt_a100_3gpu_coco_log_rect_seg.sbatch](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/cluster/stt_a100_3gpu_coco_log_rect_seg.sbatch)

These are parallel to the existing ring-based COCO launchers, but point to:
- the new log-rect configs
- the new log-rect output roots
- the log-rect MAE exported encoder artifact

## 7. W&B Change

We changed the W&B project name to:
- `SegmentThisLogRectilinearZaratan`

This replaced the old project naming in runtime/config usage so the rectilinear work is tracked separately.

Relevant file:
- [stt_pipeline/config.py](/Users/davitsoselia/Downloads/SegmentThisLogRectilinear/stt_pipeline/config.py)

## 8. What We Deliberately Did Not Change

To keep the comparison fair, we did not change:
- STT backbone size
- decoder architecture
- COCO dataset/manifests
- train/eval task definitions
- MAE objective
- segmentation losses
- optimizer family
- warmup/schedule structure
- number of GPUs

So this experiment is not “new model + new tokenizer.”
It is essentially:
- same training recipe
- same dataset
- same cluster setup
- different tokenization geometry

## 9. Smoke and Real Runs Completed

### Smoke validation

Passed:
- local benchmark
- local preflight
- local MAE smoke
- local segmentation-start smoke
- server benchmark on CUDA
- server preflight on CUDA
- server MAE smoke
- server segmentation smoke from MAE artifact

### Real MAE run

Completed successfully:
- profile: `stt_b_coco_log_rect_mae_3xa100`
- exported artifact:
  - `/home/dsoselia/scratch.varshney-prj/SegmentThisThingLogRect/artifacts/stt_b_coco_log_rect_mae_3xa100_final_encoder.pt`

W&B:
- synced online under project `SegmentThisLogRectilinearZaratan`

### Real segmentation run

Started successfully and reached:
- step `500`
- first checkpoint written:
  - `checkpoint_step_0000500.pt`

Then failed during the first distributed eval with:
- NCCL `ALLREDUCE` timeout

Observed behavior:
- training itself ran through step 500
- failure happened after checkpointing, in the eval/distributed phase
- `RUN_STATUS.md` remained stale at `state: running`, but Slurm log shows the real crash

## 10. Current Outcome So Far

### Functional outcome

Successes:
- log-rectilinear tokenizer is fully integrated into the repo’s training stack
- MAE pretraining works end-to-end
- segmentation training works at least through the first checkpoint
- W&B logging path works in offline mode and can be synced from the login node

### Performance outcome so far

Compared with the original ring STT COCO segmentation run over the same early training window:
- log-rect was slower
- no memory win was observed

Early comparison:
- log-rect: about `5.66 s/step`, `362 samples/s`
- ring baseline: about `3.93 s/step`, `521.5 samples/s`

Interpretation:
- despite slightly fewer tokens (`169` vs `172`), the current log-rect path is slower in practice
- likely overhead is outside pure token count, e.g. foveation/build/projection path or eval/training plumbing

## 11. Slide-Friendly Short Version

### One-line summary

We replaced STT’s ring tokenizer with a log-rectilinear box tokenizer while keeping the rest of the COCO training recipe fixed.

### Minimal bullet version

- Added a second tokenizer path: `log_rect_box`
- Integrated `LogRectilinearFoveator` into train/eval/benchmark code
- Matched the baseline crop size (`1280`) and nearly matched token count (`169` vs `172`)
- Added dedicated COCO MAE + segmentation configs and 3xA100 launchers
- Forced segmentation to initialize from log-rect MAE output, not the released ring STT checkpoint
- Renamed W&B project to `SegmentThisLogRectilinearZaratan`
- Fixed manifest path resolution so server-side fresh clones work
- MAE completed successfully
- segmentation reached first checkpoint, then hit an NCCL timeout during distributed eval
- early training speed is currently worse than the original STT ring baseline

## 12. Suggested Slide Titles

- `What We Changed for Log-Rectilinear STT`
- `Tokenizer-Only Modification to STT`
- `Code Changes: Ring Foveation to Log-Rect Box`
- `Experiment Parity with Baseline COCO STT`
- `Current Status and Early Findings`
