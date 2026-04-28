from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.flatten(1).mean(dim=1)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prob = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (prob * targets).sum(dim=1)
    denominator = prob.sum(dim=1) + targets.sum(dim=1)
    return 1 - (numerator + 1) / (denominator + 1)


def _actual_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = logits.sigmoid() > 0.5
    target = targets > 0.5
    inter = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def multimask_segmentation_loss(
    pred_masks: torch.Tensor,
    pred_iou: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    focal_weight: float,
    dice_weight: float,
    iou_weight: float,
    focal_alpha: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_masks.ndim == 3:
        target_masks = target_masks.unsqueeze(1)
    num_masks = pred_masks.shape[1]
    expanded_targets = target_masks.expand(-1, num_masks, -1, -1)
    focal = sigmoid_focal_loss(
        pred_masks.flatten(0, 1),
        expanded_targets.flatten(0, 1),
        alpha=focal_alpha,
        gamma=focal_gamma,
    ).view(pred_masks.shape[0], num_masks)
    dice = dice_loss(pred_masks.flatten(0, 1), expanded_targets.flatten(0, 1)).view(pred_masks.shape[0], num_masks)
    seg_loss = focal_weight * focal + dice_weight * dice
    best_idx = seg_loss.argmin(dim=1)
    batch_idx = torch.arange(pred_masks.shape[0], device=pred_masks.device)
    chosen_masks = pred_masks[batch_idx, best_idx]
    chosen_iou_pred = pred_iou[batch_idx, best_idx]
    chosen_seg_loss = seg_loss[batch_idx, best_idx]
    actual_iou = _actual_iou(chosen_masks, target_masks[:, 0])
    iou_loss = F.mse_loss(chosen_iou_pred, actual_iou, reduction="none")
    total_loss = chosen_seg_loss + iou_weight * iou_loss
    metrics = {
        "loss": float(total_loss.mean().detach().cpu()),
        "seg_loss": float(chosen_seg_loss.mean().detach().cpu()),
        "focal_loss": float(focal[batch_idx, best_idx].mean().detach().cpu()),
        "dice_loss": float(dice[batch_idx, best_idx].mean().detach().cpu()),
        "iou_loss": float(iou_loss.mean().detach().cpu()),
        "actual_iou": float(actual_iou.mean().detach().cpu()),
    }
    return total_loss.mean(), metrics
