from __future__ import annotations

import torch


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    numerator = 2.0 * (probs * target).sum(dim=(-1, -2, -3))
    denominator = probs.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
    return 1.0 - (numerator + eps) / (denominator + eps)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = logits.sigmoid()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - pt).pow(gamma) * ce
    return loss.mean(dim=(-1, -2, -3))


def expected_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = logits.sigmoid()
    intersection = (probs * target).sum(dim=(-1, -2, -3))
    union = (probs + target - probs * target).sum(dim=(-1, -2, -3))
    return (intersection + eps) / (union + eps)


def multimask_segmentation_loss(
    pred_masks: torch.Tensor,
    pred_iou: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float,
    dice_weight: float,
    iou_weight: float,
    focal_alpha: float,
    focal_gamma: float,
) -> tuple[torch.Tensor, dict]:
    if target.ndim == 5 and target.shape[2] == 1:
        target = target.squeeze(2)
    target = target.unsqueeze(1).expand(-1, pred_masks.shape[1], -1, -1, -1)
    focal = sigmoid_focal_loss(pred_masks, target, alpha=focal_alpha, gamma=focal_gamma)
    dice = soft_dice_loss(pred_masks, target)
    target_iou = expected_iou(pred_masks.detach(), target)
    iou_reg = torch.nn.functional.mse_loss(pred_iou, target_iou, reduction="none")
    total = focal_weight * focal + dice_weight * dice + iou_weight * iou_reg
    best_idx = total.argmin(dim=1)
    batch_indices = torch.arange(pred_masks.shape[0], device=pred_masks.device)
    loss = total[batch_indices, best_idx].mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "focal_loss": float(focal[batch_indices, best_idx].mean().detach().cpu()),
        "dice_loss": float(dice[batch_indices, best_idx].mean().detach().cpu()),
        "iou_loss": float(iou_reg[batch_indices, best_idx].mean().detach().cpu()),
        "soft_iou": float(target_iou[batch_indices, best_idx].mean().detach().cpu()),
    }
    return loss, metrics
