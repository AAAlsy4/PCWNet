from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def prompt_weighted_pool(
    feature: torch.Tensor, prompt_map: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Pool ``[B,C,H,W]`` features using a prompt map, returning ``[B,C]``."""
    if prompt_map.ndim == 3:
        prompt_map = prompt_map.unsqueeze(1)
    if prompt_map.ndim != 4 or prompt_map.shape[1] != 1:
        raise ValueError("prompt_map must have shape [B,H,W] or [B,1,H,W]")
    weights = F.interpolate(
        prompt_map.float(), size=feature.shape[-2:], mode="bilinear", align_corners=False
    ).clamp_min(0)
    weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (feature * weights.to(feature.dtype)).sum(dim=(-2, -1))


def local_soft_argmax_2d(
    logits: torch.Tensor, topk: int, radius: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode top modes from ``[B,H,W]`` logits into centers, indices, and scores."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B,H,W]")
    batch, height, width = logits.shape
    mode_count = min(topk, height * width)
    if mode_count < 1:
        raise ValueError("topk must be positive")

    kernel = 2 * radius + 1
    pooled = F.max_pool2d(
        logits.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius
    ).squeeze(1)
    peak_logits = logits.masked_fill(logits < pooled, float("-inf"))
    peak_values, peak_indices = peak_logits.flatten(1).topk(mode_count, dim=1)

    invalid = ~torch.isfinite(peak_values)
    if invalid.any():
        fallback_values, fallback_indices = logits.flatten(1).topk(mode_count, dim=1)
        peak_indices = torch.where(invalid, fallback_indices, peak_indices)

    center_y = torch.div(peak_indices, width, rounding_mode="floor")
    center_x = peak_indices.remainder(width)
    offsets = torch.arange(-radius, radius + 1, device=logits.device)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_y = offset_y.flatten().view(1, 1, -1)
    offset_x = offset_x.flatten().view(1, 1, -1)

    sample_y = center_y.unsqueeze(-1) + offset_y
    sample_x = center_x.unsqueeze(-1) + offset_x
    valid = (
        (sample_y >= 0)
        & (sample_y < height)
        & (sample_x >= 0)
        & (sample_x < width)
    )
    clamped_y = sample_y.clamp(0, height - 1)
    clamped_x = sample_x.clamp(0, width - 1)
    local_indices = clamped_y * width + clamped_x
    local_logits = torch.gather(
        logits.flatten(1).unsqueeze(1).expand(-1, mode_count, -1),
        2,
        local_indices.expand(batch, -1, -1),
    ).masked_fill(~valid.expand(batch, -1, -1), float("-inf"))
    local_weights = local_logits.softmax(dim=-1)

    x_coords = (clamped_x.to(logits.dtype) + 0.5) / width
    y_coords = (clamped_y.to(logits.dtype) + 0.5) / height
    refined_x = (local_weights * x_coords).sum(dim=-1)
    refined_y = (local_weights * y_coords).sum(dim=-1)
    centers = torch.stack((refined_x, refined_y), dim=-1)

    probabilities = logits.flatten(1).softmax(dim=-1)
    mode_scores = torch.gather(probabilities, 1, peak_indices)
    return centers, peak_indices, mode_scores


def sample_candidate_patches(
    feature: torch.Tensor, states: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Crop candidate-aligned patches and return ``[B,K,C,S,S]`` features."""
    batch, channels, _, _ = feature.shape
    candidate_count = states.shape[1]
    centers = states[..., :2]
    sizes = states[..., 2:4].exp()
    axis = torch.linspace(
        -0.5, 0.5, output_size, device=feature.device, dtype=feature.dtype
    )
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1).view(
        1, 1, output_size, output_size, 2
    )
    grid = centers[:, :, None, None] + base_grid * sizes[:, :, None, None]
    grid = grid.mul(2).sub(1).reshape(
        batch * candidate_count, output_size, output_size, 2
    )
    expanded_feature = feature[:, None].expand(
        -1, candidate_count, -1, -1, -1
    ).reshape(batch * candidate_count, channels, *feature.shape[-2:])
    patches = F.grid_sample(
        expanded_feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return patches.view(
        batch, candidate_count, channels, output_size, output_size
    )


def states_to_boxes(states: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``(cx, cy, log_w, log_h)`` states to XYXY boxes."""
    centers, sizes = states[..., :2], states[..., 2:4].exp()
    half_sizes = sizes / 2
    top_left = centers - half_sizes
    bottom_right = centers + half_sizes
    return torch.cat((top_left, bottom_right), dim=-1).clamp(0.0, 1.0)
