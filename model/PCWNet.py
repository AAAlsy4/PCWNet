from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import prompt_weighted_pool, local_soft_argmax_2d, sample_candidate_patches, states_to_boxes


TensorDict = Dict[str, torch.Tensor]


@dataclass
class PCWNetConfig:
    """Hyperparameters controlling the PCWNet model architecture."""

    pretrained_backbones: bool = True
    freeze_coarse: bool = True
    decoder_dim: int = 512
    decoder_heads: int = 8
    decoder_layers: int = 5
    decoder_ffn_dim: int = 2048
    anchor_grid_size: Tuple[int, int] = (32, 32)
    topk: int = 5
    anchor_score_power: float = 1.0
    local_softargmax_radius: int = 1
    refinement_levels: Tuple[str, ...] = ("layer3", "layer2", "layer1")
    refinement_dim: int = 256
    refinement_patch_size: int = 7
    min_box_size: float = 1.0 / 1024.0
    max_box_size: float = 1.0


class CoarseSemanticEncoder(nn.Module):
    """Shared supervised ConvNeXt encoder for cross-view candidate search."""

    out_channels = 768

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.features = models.convnext_tiny(weights=weights).features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 3, H, W]`` images into a coarse feature map."""
        return self.features(images)


class FineFeaturePyramid(nn.Module):
    """View-specific ResNet34 pyramid for object-level warp refinement."""

    out_channels = {"layer1": 64, "layer2": 128, "layer3": 256}

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

    def forward(self, images: torch.Tensor) -> TensorDict:
        """Return multi-scale feature maps for ``[B, 3, H, W]`` images."""
        x = self.stem(images)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
        }





class PromptConditionedAnchorDecoder(nn.Module):
    """Position-agnostic transformer decoder for target anchor probabilities."""

    def __init__(
        self,
        dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        self.prompt_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.reference_type = nn.Parameter(torch.zeros(1, 1, dim))
        self.anchor_head = nn.Linear(dim, 1)
        self.size_head = nn.Linear(dim, 2)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        nn.init.normal_(self.prompt_type, std=0.02)
        nn.init.normal_(self.reference_type, std=0.02)

    def forward(
        self, prompt_token: torch.Tensor, reference_tokens: torch.Tensor
    ) -> TensorDict:
        """Decode prompt and reference tokens into anchor and size logits."""
        prompt = prompt_token.unsqueeze(1) + self.prompt_type
        reference = reference_tokens + self.reference_type
        encoded = self.transformer(torch.cat((prompt, reference), dim=1))
        prompt_context, reference_context = encoded[:, 0], encoded[:, 1:]

        similarity = torch.einsum(
            "bd,bnd->bn",
            F.normalize(prompt_context, dim=-1),
            F.normalize(reference_context, dim=-1),
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        anchor_logits = self.anchor_head(reference_context).squeeze(-1)
        anchor_logits = anchor_logits + scale * similarity
        return {
            "anchor_logits_flat": anchor_logits,
            "size_logits_flat": self.size_head(reference_context),
        }





class ObjectWarpRefiner(nn.Module):
    """RoMa-style local refiner whose state is an object box, not dense flow."""

    def __init__(
        self,
        query_channels: int,
        reference_channels: int,
        hidden_dim: int,
        patch_size: int,
        center_step: float = 0.75,
        size_step: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.center_step = center_step
        self.size_step = size_step
        self.query_proj = nn.Linear(query_channels, hidden_dim)
        self.reference_proj = nn.Conv2d(reference_channels, hidden_dim, 1)
        mlp_input = hidden_dim * 2 + patch_size * patch_size + 5
        self.update = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.state_head = nn.Linear(hidden_dim, 4)
        self.certainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        query_token: torch.Tensor,
        reference_feature: torch.Tensor,
        states: torch.Tensor,
        certainty_logits: torch.Tensor,
        min_box_size: float,
        max_box_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine candidate box states and return states, certainty, and correlation."""
        patches = sample_candidate_patches(
            reference_feature, states, self.patch_size
        )
        batch, candidate_count, channels, patch_h, patch_w = patches.shape
        projected_patches = self.reference_proj(
            patches.reshape(batch * candidate_count, channels, patch_h, patch_w)
        ).view(batch, candidate_count, -1, patch_h, patch_w)
        projected_query = self.query_proj(query_token)
        query = projected_query[:, None].expand(-1, candidate_count, -1)

        correlations = (
            F.normalize(projected_patches, dim=2)
            * F.normalize(query, dim=-1)[..., None, None]
        ).sum(dim=2)
        pooled_reference = projected_patches.mean(dim=(-2, -1))
        update_input = torch.cat(
            (
                query,
                pooled_reference,
                correlations.flatten(2),
                states,
                certainty_logits.unsqueeze(-1),
            ),
            dim=-1,
        )
        hidden = self.update(update_input)
        raw_delta = self.state_head(hidden)
        current_size = states[..., 2:4].exp()
        delta_center = (
            raw_delta[..., :2].tanh()
            * current_size
            * self.center_step
        )
        delta_size = raw_delta[..., 2:4].tanh() * self.size_step
        centers = (states[..., :2] + delta_center).clamp(0.0, 1.0)
        log_sizes = states[..., 2:4] + delta_size
        log_sizes = log_sizes.clamp(
            min=math.log(min_box_size), max=math.log(max_box_size)
        )
        new_states = torch.cat((centers, log_sizes), dim=-1)
        new_certainty = certainty_logits + self.certainty_head(hidden).squeeze(-1)
        return new_states, new_certainty, correlations





class PCWNet(nn.Module):
    """Probabilistic Coarse-to-fine Warp Network"""

    def __init__(self, config: Optional[PCWNetConfig] = None) -> None:
        super().__init__()
        self.config = config or PCWNetConfig()
        cfg = self.config

        self.coarse_encoder = CoarseSemanticEncoder(cfg.pretrained_backbones)
        self.query_fine_encoder = FineFeaturePyramid(
            cfg.pretrained_backbones
        )
        self.reference_fine_encoder = FineFeaturePyramid(
            cfg.pretrained_backbones
        )

        self.query_coarse_proj = nn.Linear(
            self.coarse_encoder.out_channels, cfg.decoder_dim
        )
        self.reference_coarse_proj = nn.Conv2d(
            self.coarse_encoder.out_channels, cfg.decoder_dim, kernel_size=1
        )
        self.anchor_decoder = PromptConditionedAnchorDecoder(
            dim=cfg.decoder_dim,
            heads=cfg.decoder_heads,
            layers=cfg.decoder_layers,
            ffn_dim=cfg.decoder_ffn_dim,
        )

        fine_channels = FineFeaturePyramid.out_channels
        self.refiners = nn.ModuleDict(
            {
                level: ObjectWarpRefiner(
                    query_channels=fine_channels[level],
                    reference_channels=fine_channels[level],
                    hidden_dim=cfg.refinement_dim,
                    patch_size=cfg.refinement_patch_size,
                )
                for level in cfg.refinement_levels
            }
        )

        if cfg.freeze_coarse:
            for parameter in self.coarse_encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "PCWNet":
        """Set training mode while keeping configured frozen encoders in eval mode."""
        super().train(mode)
        if self.config.freeze_coarse:
            self.coarse_encoder.eval()
        return self

    def _encode_coarse(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with gradients disabled when the coarse encoder is frozen."""
        if self.config.freeze_coarse:
            with torch.no_grad():
                return self.coarse_encoder(images)
        return self.coarse_encoder(images)

    def forward(
        self,
        query_images: torch.Tensor,
        reference_images: torch.Tensor,
        prompt_map: torch.Tensor,
    ) -> TensorDict:
        """Register prompted query objects in reference images.

        Args:
            query_images: Query images shaped ``[B, 3, H, W]``.
            reference_images: Reference images shaped ``[B, 3, H, W]``.
            prompt_map: Query point prompts shaped ``[B, H, W]`` or ``[B, 1, H, W]``.

        Returns:
            A dictionary containing normalized boxes, confidence scores, candidate
            predictions, and intermediate logits used by the training criterion.
        """
        cfg = self.config
        if query_images.shape[0] != reference_images.shape[0]:
            raise ValueError("query and reference batch sizes must match")

        query_coarse = self._encode_coarse(query_images)
        reference_coarse = self._encode_coarse(reference_images)
        query_fine = self.query_fine_encoder(query_images)
        reference_fine = self.reference_fine_encoder(reference_images)

        prompt_coarse = prompt_weighted_pool(query_coarse, prompt_map)
        prompt_token = self.query_coarse_proj(prompt_coarse)
        reference_grid = F.adaptive_avg_pool2d(
            reference_coarse, cfg.anchor_grid_size
        )
        reference_grid = self.reference_coarse_proj(reference_grid)
        batch, channels, anchor_h, anchor_w = reference_grid.shape
        reference_tokens = reference_grid.flatten(2).transpose(1, 2)

        decoded = self.anchor_decoder(prompt_token, reference_tokens)
        anchor_logits = decoded["anchor_logits_flat"].view(
            batch, anchor_h, anchor_w
        )
        centers, anchor_indices, _ = local_soft_argmax_2d(
            anchor_logits,
            topk=cfg.topk,
            radius=cfg.local_softargmax_radius,
        )
        anchor_probabilities_flat = decoded["anchor_logits_flat"].softmax(dim=-1)
        anchor_scores = torch.gather(
            anchor_probabilities_flat, 1, anchor_indices
        )

        size_logits = torch.gather(
            decoded["size_logits_flat"],
            1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, 2),
        )
        initial_sizes = cfg.min_box_size + size_logits.sigmoid() * (
            cfg.max_box_size - cfg.min_box_size
        )
        states = torch.cat((centers, initial_sizes.log()), dim=-1)
        certainty_logits = torch.zeros_like(anchor_scores)

        refinement_states: List[torch.Tensor] = [states]
        refinement_certainties: List[torch.Tensor] = [certainty_logits]
        local_correlations: List[torch.Tensor] = []
        for level in cfg.refinement_levels:
            query_token = prompt_weighted_pool(query_fine[level], prompt_map)
            states, certainty_logits, correlations = self.refiners[level](
                query_token,
                reference_fine[level],
                states,
                certainty_logits,
                min_box_size=cfg.min_box_size,
                max_box_size=cfg.max_box_size,
            )
            refinement_states.append(states)
            refinement_certainties.append(certainty_logits)
            local_correlations.append(correlations)

        candidate_boxes = states_to_boxes(states)
        anchor_score_power = max(cfg.anchor_score_power, 0.0)
        candidate_selection_logits = (
            anchor_score_power * anchor_scores.clamp_min(1e-8).log()
            + F.logsigmoid(certainty_logits)
        )
        candidate_scores = candidate_selection_logits.exp()
        selected_indices = candidate_scores.argmax(dim=1)
        batch_indices = torch.arange(batch, device=query_images.device)
        boxes = candidate_boxes[batch_indices, selected_indices]
        scores = candidate_scores[batch_indices, selected_indices]

        return {
            **decoded,
            "anchor_logits": anchor_logits,
            "anchor_probabilities": anchor_probabilities_flat.view(
                batch, anchor_h, anchor_w
            ),
            "candidate_anchor_indices": anchor_indices,
            "candidate_anchor_scores": anchor_scores,
            "candidate_centers": centers,
            "refinement_states": refinement_states,
            "refinement_certainties": refinement_certainties,
            "local_correlations": local_correlations,
            "candidate_boxes": candidate_boxes,
            "candidate_certainty_logits": certainty_logits,
            "candidate_selection_logits": candidate_selection_logits,
            "candidate_scores": candidate_scores,
            "selected_indices": selected_indices,
            "boxes": boxes,
            "scores": scores,
        }
