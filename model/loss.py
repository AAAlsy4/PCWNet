from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    """Weights and robust-loss parameters used by ``Criterion``."""

    anchor_weight: float = 1.0
    warp_weight: float = 2.0
    box_l1_weight: float = 2.0
    giou_weight: float = 2.0
    certainty_weight: float = 0.25
    robust_alpha: float = 0.5
    robust_scale: float = 0.03


def normalize_xyxy_boxes(
    boxes: torch.Tensor, image_size: Sequence[int]
) -> torch.Tensor:
    """Normalize pixel-space XYXY boxes to the ``[0, 1]`` image coordinate frame."""
    height, width = int(image_size[0]), int(image_size[1])
    scale = boxes.new_tensor((width, height, width, height))
    return (boxes / scale).clamp(0.0, 1.0)


def boxes_to_states(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert normalized XYXY boxes to ``(cx, cy, log_w, log_h)`` states."""
    top_left, bottom_right = boxes[..., :2], boxes[..., 2:]
    sizes = (bottom_right - top_left).clamp_min(eps)
    centers = (top_left + bottom_right) / 2
    return torch.cat((centers, sizes.log()), dim=-1)


def aligned_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute element-wise IoU for aligned normalized XYXY box tensors."""
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-7)


def aligned_giou_loss(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return element-wise generalized IoU loss for aligned XYXY boxes."""
    iou = aligned_iou(boxes1, boxes2)
    enclosing_top_left = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    enclosing_bottom_right = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_area = (
        enclosing_bottom_right - enclosing_top_left
    ).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    top_left = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    bottom_right = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    union = area1 + area2 - intersection
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-7)
    return 1 - giou


def robust_charbonnier(
    error: torch.Tensor, alpha: float, scale: float
) -> torch.Tensor:
    """Apply a smooth robust penalty to an arbitrary error tensor."""
    normalized = error / scale
    return (scale**alpha) * (normalized.square() + 1.0).pow(alpha / 2)


class Criterion(nn.Module):
    """Combine coarse matching, box refinement, and candidate ranking losses."""

    def __init__(self, config: Optional[LossConfig] = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        outputs: Dict[str, object],
        target_boxes_xyxy: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses and selection metrics.

        Args:
            outputs: Output dictionary produced by ``PCWNet.forward``.
            target_boxes_xyxy: Ground-truth boxes in pixel-space XYXY format.
            image_size: Reference image ``(height, width)``.
            refinement_weight_scale: Multiplier for refinement-related losses.

        Returns:
            A dictionary containing the total loss, component losses, and metrics.
        """
        cfg = self.config
        target_boxes = normalize_xyxy_boxes(target_boxes_xyxy, image_size)
        target_states = boxes_to_states(target_boxes)
        device = target_boxes.device

        anchor_logits = outputs["anchor_logits"]
        anchor_h, anchor_w = anchor_logits.shape[-2:]
        center_x = (target_states[:, 0] * anchor_w).long().clamp(0, anchor_w - 1)
        center_y = (target_states[:, 1] * anchor_h).long().clamp(0, anchor_h - 1)
        target_anchor = center_y * anchor_w + center_x
        anchor_loss = F.cross_entropy(
            outputs["anchor_logits_flat"],
            target_anchor,
        )

        warp_loss = anchor_loss * 0.0
        refinement_states = outputs["refinement_states"]
        for state in refinement_states[1:]:
            center_distance = (
                state[..., :2] - target_states[:, None, :2]
            ).square().sum(dim=-1)
            best_index = center_distance.argmin(dim=1)
            rows = torch.arange(state.shape[0], device=device)
            best_state = state[rows, best_index]
            state_error = best_state - target_states
            warp_loss = warp_loss + robust_charbonnier(
                state_error, cfg.robust_alpha, cfg.robust_scale
            ).mean()
        warp_loss = warp_loss / max(len(refinement_states) - 1, 1)

        candidate_boxes = outputs["candidate_boxes"]
        expanded_targets = target_boxes[:, None].expand_as(candidate_boxes)
        candidate_ious = aligned_iou(candidate_boxes, expanded_targets)
        best_index = candidate_ious.argmax(dim=1)
        rows = torch.arange(candidate_boxes.shape[0], device=device)
        best_boxes = candidate_boxes[rows, best_index]
        box_l1_loss = F.l1_loss(best_boxes, target_boxes)
        giou_loss = aligned_giou_loss(best_boxes, target_boxes).mean()
        certainty_logits = outputs["candidate_certainty_logits"]
        certainty_loss = F.binary_cross_entropy_with_logits(
            certainty_logits, candidate_ious.detach()
        )
        total = (
            cfg.anchor_weight * anchor_loss
            + cfg.box_l1_weight * box_l1_loss
            + cfg.giou_weight * giou_loss
            + cfg.warp_weight * warp_loss
            + cfg.certainty_weight * certainty_loss
        )
        return {
            "loss": total,
            "loss_anchor": anchor_loss,
            "loss_warp": warp_loss,
            "loss_box_l1": box_l1_loss,
            "loss_giou": giou_loss,
            "loss_certainty": certainty_loss,
        }
