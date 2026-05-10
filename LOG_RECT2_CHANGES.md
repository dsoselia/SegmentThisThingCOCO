# LogRect2 COCO Experiment

This branch contains the second Zaratan COCO log-rectilinear STT attempt. It keeps
the existing `log_rect_box` tokenizer interface, but replaces the weak/near-uniform
axis spacing with the true log-rectilinear crop mapping used in the Nexus-style
implementation.

Default tokenizer settings:

- `token_size`: 16
- `pattern_size`: 1280
- `log_rect_axis_bins`: 13
- `log_rect_exponent`: 4.0
- `log_rect_center_width`: 16
- token count: 169

The resulting 1D crop-space bin widths are approximately:

```text
[390, 153, 41, 16, 16, 16, 16, 16, 16, 16, 40, 153, 391]
```

The model-facing buffer remains compatible with STT:

```text
169 x 3 x 16 x 16
```

Interpreted as a packed image-like buffer, this corresponds to `208 x 208`
samples over a prompt-centered `1280 x 1280` crop.

The COCO configs point to the existing Zaratan COCO manifests under the STT
experiment tree and use offline W&B project `SegmentThisThingLogRect2Zaratan`.
