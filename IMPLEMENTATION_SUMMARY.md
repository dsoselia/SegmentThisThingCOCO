# Segment This Thing Replication Summary

## Project Overview

This project is a best-effort public reconstruction of the training and evaluation pipeline for Meta's Segment This Thing (STT), built on top of the released inference-only repository. The released repo provides the model architecture, foveation code, predictor, and pretrained checkpoints. The missing pieces for replication were the training stack, dataset plumbing, evaluation harness, and experiment orchestration.

The implemented work adds those missing layers while preserving the original `segment_this_thing` package as the source of truth for the core model and foveation logic.

## Method Summary

STT is a point-prompted segmentation model that replaces uniform patch tokenization with foveated tokenization. Instead of encoding the full image at a single fixed spatial resolution, it crops around the prompt and allocates higher-resolution tokens near the center and lower-resolution tokens toward the periphery. This reduces token count and compute while preserving more detail where the prompt matters most.

The implemented replication stack supports:

- foveated MAE-style encoder pretraining
- segmentation fine-tuning with continuous foveated targets
- expected-IoU supervision for the IoU head
- multimask best-of-K training behavior
- manifest-driven evaluation with mask reprojection back to image space

## Current Implementation

The added components are:

- `stt_pipeline/`
  - config loading
  - runtime/environment helpers
  - model builders/checkpoint IO
  - manifest-driven datasets
  - foveated transforms and reprojection
  - segmentation losses
  - MAE pretraining module
  - training loops
  - evaluation
- `scripts/run_stt.py`
  - `pretrain-mae`
  - `train-stt`
  - `eval-stt`
  - `benchmark-stt`
- `scripts/make_sa1b_subset_manifest.py`
  - builds train/val manifests directly from extracted SA-1B shard JSONs
- `configs/`
  - smoke config
  - paper-style template config
  - SA-1B proxy config

The core model package now also contains a second tokenizer path:

- `segment_this_thing.LogRectilinearFoveator`
  - explicit per-token rectangular boxes instead of nested stride rings
  - log-rectilinear spacing with summed-area-table averaging
  - matched-budget default of `13 x 13 = 169` tokens for `STT-B`
  - integrated into training, eval, and benchmarking via `model.tokenizer_type = "log_rect_box"`

## Assumptions and Deviations

- The public repo does not include the original internal training code, so the current pipeline is a reconstruction rather than an exact reproduction.
- Prompt noise is disabled by default because the paper describes it as optional but does not publish the noise scale.
- The MAE stage is best-effort and may differ from the authors' original internal implementation details.
- The log-rectilinear tokenizer prototype is treated as a new tokenizer variant and is not checkpoint-compatible with the released STT weights.
- Real SA-1B manifests are generated from extracted shard JSONs and use the shard-provided COCO-style RLE masks directly.
- Proxy evaluation on held-out SA-1B is used as a short-run signal when the paper's 9 public evaluation datasets are not staged.

## Validation and Results So Far

Completed validations:

- local smoke validation for benchmark, MAE pretraining, segmentation training, and eval
- local smoke validation for the new `log_rect_box` tokenizer path
- cluster smoke validation on `zaratan-compute` with `py12`
- real SA-1B shard staging and manifest generation
- held-out SA-1B proxy evaluation of the released STT-B checkpoint
- single-H100 short fine-tune run initialized from the released STT-B checkpoint

Current tokenizer-prototype results:

- `LogRectilinearFoveator` builds a `169`-token pattern on the same `1280` crop size used by baseline `STT-B`
- the central token remains one-to-one at `16 x 16` pixels
- local benchmark smoke on CPU:
  - baseline `stt_ring`: about `75.5 ms`
  - prototype `log_rect_box`: about `106.5 ms`
- smoke MAE, segmentation training, and eval all complete end to end with the new tokenizer

Current proxy results:

- released STT-B checkpoint on held-out SA-1B proxy set: about `0.6893` mIoU over `300` examples
- short-run fine-tuned checkpoint at step `250`: about `0.6728` mIoU over the same proxy set

Observed single-H100 training behavior:

- effective batch size `32`
- post-startup step time roughly `0.57s` to `0.73s`
- GPU memory around `2.8 GB`
- losses and soft IoU move in plausible ranges during training

## Current Dataset State

- One SA-1B shard has been fully staged and extracted:
  - `sa_000020.tar`
  - about `11G` compressed
  - about `12G` apparent extracted size
  - `11186` images and `11186` JSON files
- Additional shard staging toward a larger subset is in progress from the login node.

## Limitations

- The current proxy run is not comparable to the paper's published benchmark table.
- Only a small SA-1B subset is currently staged, far below full SA-1B scale.
- The full 9-dataset evaluation from the paper is not yet wired into the cluster workflow.
- The training job is currently more data-loader bound than memory bound on a single H100.

## Next Practical Steps

- finish staging more SA-1B shards toward the subset storage target
- regenerate manifests from the expanded subset
- let the single-H100 proxy run continue to produce later checkpoints
- evaluate later checkpoints against the same held-out proxy split
- stage the paper's external evaluation datasets for a more meaningful comparison
