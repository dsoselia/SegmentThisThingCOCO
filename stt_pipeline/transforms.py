from __future__ import annotations

from typing import Optional

import torch

from segment_this_thing.utils import get_centered_crop, get_crop_bounds


def get_centered_mask_crop(mask: torch.Tensor, crop_bounds: torch.Tensor) -> torch.Tensor:
    device = mask.device
    lower_corner, upper_corner = crop_bounds
    lower_pad = (-lower_corner).clamp(min=0)
    upper_pad = (upper_corner - torch.tensor(mask.shape[1::-1], device=device)).clamp(min=0)

    lower_corner = lower_corner + lower_pad
    upper_corner = upper_corner - upper_pad
    crop = mask[lower_corner[1] : upper_corner[1], lower_corner[0] : upper_corner[0]]

    if lower_pad[0] > 0:
        crop = torch.cat([torch.zeros((crop.shape[0], lower_pad[0]), device=device), crop], dim=1)
    if upper_pad[0] > 0:
        crop = torch.cat([crop, torch.zeros((crop.shape[0], upper_pad[0]), device=device)], dim=1)
    if lower_pad[1] > 0:
        crop = torch.cat([torch.zeros((lower_pad[1], crop.shape[1]), device=device), crop], dim=0)
    if upper_pad[1] > 0:
        crop = torch.cat([crop, torch.zeros((upper_pad[1], crop.shape[1]), device=device)], dim=0)

    return crop


def _compute_integral_map(values: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(values, (1, 0, 1, 0), mode="constant", value=0.0)
    return padded.cumsum(dim=-2).cumsum(dim=-1)


def extract_scalar_foveation(foveator, scalar_image: torch.Tensor) -> torch.Tensor:
    if scalar_image.ndim != 2:
        raise ValueError(f"Expected scalar image with shape (H, W), got {tuple(scalar_image.shape)}")
    device = scalar_image.device
    integral = _compute_integral_map(scalar_image)
    if hasattr(foveator, "get_bin_coordinates"):
        lower, upper, area = foveator.get_bin_coordinates()
        lower = lower.to(device)
        upper = upper.to(device)
        area = area.to(device).float()
        top_right = torch.stack([upper[..., 0], lower[..., 1]], dim=-1)
        bottom_left = torch.stack([lower[..., 0], upper[..., 1]], dim=-1)
        summed = (
            foveator._sample_integral(integral.unsqueeze(0), upper).squeeze(0)
            - foveator._sample_integral(integral.unsqueeze(0), top_right).squeeze(0)
            - foveator._sample_integral(integral.unsqueeze(0), bottom_left).squeeze(0)
            + foveator._sample_integral(integral.unsqueeze(0), lower).squeeze(0)
        )
        return (summed / area).unsqueeze(1)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(foveator.token_size, device=device),
            torch.arange(foveator.token_size, device=device),
            indexing="xy",
        ),
        dim=-1,
    )
    lower = foveator.token_corner_indices.to(device).view(-1, 1, 1, 2) + foveator.token_strides.to(device).view(-1, 1, 1, 1) * grid.unsqueeze(0)
    upper = lower + foveator.token_strides.to(device).view(-1, 1, 1, 1)
    area = foveator.token_strides.to(device).float().square().view(-1, 1, 1)
    summed = (
        integral[upper[..., 1], upper[..., 0]]
        - integral[upper[..., 1], lower[..., 0]]
        - integral[lower[..., 1], upper[..., 0]]
        + integral[lower[..., 1], lower[..., 0]]
    )
    return (summed / area).unsqueeze(1)


def project_mask_to_foveation(foveator, mask: torch.Tensor, center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    crop_bounds = get_crop_bounds(center, foveator.get_pattern_bounds_size()).to(mask.device)
    crop = get_centered_mask_crop(mask.float(), crop_bounds)
    tokens = extract_scalar_foveation(foveator, crop)
    return tokens, crop_bounds


def build_model_inputs(image: torch.Tensor, center: torch.Tensor, foveator, in_bounds_threshold: float = 0.0):
    crop_bounds = get_crop_bounds(center, foveator.get_pattern_bounds_size()).to(image.device)
    crop = get_centered_crop(image, crop_bounds)
    valid_mask = foveator.get_in_bounds_tokens(
        torch.tensor(image.shape[1::-1], device=image.device),
        crop_bounds,
        in_bounds_threshold=in_bounds_threshold,
    )
    tokens = foveator.extract_foveated_image(crop.permute(2, 0, 1))
    return tokens, valid_mask, crop_bounds


def reconstruct_logits_to_image(foveator, logits: torch.Tensor, crop_bounds: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    crop = foveator.generate_foveated_visualization(logits.unsqueeze(1).cpu()).squeeze(1)
    height, width = image_size
    out = torch.zeros((crop.shape[0], height, width), dtype=crop.dtype)
    lower = crop_bounds[0].cpu()
    upper = crop_bounds[1].cpu()
    x0 = max(lower[0].item(), 0)
    y0 = max(lower[1].item(), 0)
    x1 = min(upper[0].item(), width)
    y1 = min(upper[1].item(), height)
    crop_x0 = x0 - lower[0].item()
    crop_y0 = y0 - lower[1].item()
    crop_x1 = crop_x0 + (x1 - x0)
    crop_y1 = crop_y0 + (y1 - y0)
    out[:, y0:y1, x0:x1] = crop[:, crop_y0:crop_y1, crop_x0:crop_x1]
    return out


def maybe_resize_small_image(image: torch.Tensor, min_side: int) -> torch.Tensor:
    height, width = image.shape[:2]
    if min(height, width) >= min_side:
        return image
    scale = float(min_side) / float(min(height, width))
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    resized = torch.nn.functional.interpolate(
        image.permute(2, 0, 1).unsqueeze(0).float(),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).byte()


def clamp_center_to_mask(mask: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    if mask[center[1].item(), center[0].item()]:
        return center
    ys, xs = mask.nonzero(as_tuple=True)
    if len(xs) == 0:
        return center
    coords = torch.stack([xs, ys], dim=1)
    distances = (coords - center.view(1, 2)).float().pow(2).sum(dim=1)
    return coords[distances.argmin()]
