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
    reranker: bool = False
    reranker_weight: float = 1.0
    reranker_dim: int = 128
    reranker_heads: int = 4
    reranker_layers: int = 2
    reranker_patch_size: int = 7
    query_patch_scale: float = 0.25
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
        """Build the ConvNeXt-Tiny feature extraction backbone."""
        super().__init__()
        # Coarse semantic backbone.
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.features = models.convnext_tiny(weights=weights).features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, 3, H, W]`` images into a coarse feature map."""
        return self.features(images)  # [B, 768, H/32, W/32]


class FineFeaturePyramid(nn.Module):
    """View-specific ResNet34 pyramid for object-level warp refinement."""

    out_channels = {"layer1": 64, "layer2": 128, "layer3": 256}

    def __init__(self, pretrained: bool) -> None:
        """Build the ResNet34 stem and three-level feature pyramid."""
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
        # Fine-grained feature backbone and pyramid stages.
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

    def forward(self, images: torch.Tensor) -> TensorDict:
        """Return multi-scale feature maps for ``[B, 3, H, W]`` images."""
        x = self.stem(images)  # [B, 64, H/4, W/4]
        layer1 = self.layer1(x)  # [B, 64, H/4, W/4]
        layer2 = self.layer2(layer1)  # [B, 128, H/8, W/8]
        layer3 = self.layer3(layer2)  # [B, 256, H/16, W/16]
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
        """Build the token decoder and its anchor and box-size prediction heads."""
        super().__init__()
        # Joint prompt-reference transformer encoder.
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
        # View-type embeddings and coarse prediction heads.
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
        prompt = prompt_token.unsqueeze(1) + self.prompt_type  # [B, 1, D]
        reference = reference_tokens + self.reference_type  # [B, N, D]
        encoded = self.transformer(torch.cat((prompt, reference), dim=1))  # [B, 1+N, D]
        prompt_context, reference_context = encoded[:, 0], encoded[:, 1:]  # [B, D], [B, N, D]

        similarity = torch.einsum(
            "bd,bnd->bn",
            F.normalize(prompt_context, dim=-1),
            F.normalize(reference_context, dim=-1),
        )  # [B, N]
        scale = self.logit_scale.exp().clamp(max=100.0)  # []
        anchor_logits = self.anchor_head(reference_context).squeeze(-1)  # [B, N]
        anchor_logits = anchor_logits + scale * similarity  # [B, N]
        return {
            "anchor_logits_flat": anchor_logits,
            "size_logits_flat": self.size_head(reference_context),  # [B, N, 2]
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
        """Build one local candidate-state refinement stage."""
        super().__init__()
        self.patch_size = patch_size
        self.center_step = center_step
        self.size_step = size_step
        # Query and reference feature projections.
        self.query_proj = nn.Linear(query_channels, hidden_dim)
        self.reference_proj = nn.Conv2d(reference_channels, hidden_dim, 1)
        mlp_input = hidden_dim * 2 + patch_size * patch_size + 5
        # Correlation-conditioned state update network.
        self.update = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        # Box-state and confidence prediction heads.
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
        )  # [B, K, C, S, S]
        batch, candidate_count, channels, patch_h, patch_w = patches.shape
        projected_patches = self.reference_proj(
            patches.reshape(batch * candidate_count, channels, patch_h, patch_w)
        ).view(batch, candidate_count, -1, patch_h, patch_w)  # [B, K, D, S, S]
        projected_query = self.query_proj(query_token)  # [B, D]
        query = projected_query[:, None].expand(-1, candidate_count, -1)  # [B, K, D]

        correlations = (
            F.normalize(projected_patches, dim=2)
            * F.normalize(query, dim=-1)[..., None, None]
        ).sum(dim=2)  # [B, K, S, S]
        pooled_reference = projected_patches.mean(dim=(-2, -1))  # [B, K, D]
        update_input = torch.cat(
            (
                query,
                pooled_reference,
                correlations.flatten(2),
                states,
                certainty_logits.unsqueeze(-1),
            ),
            dim=-1,
        )  # [B, K, 2D+S^2+5]
        hidden = self.update(update_input)  # [B, K, D]
        raw_delta = self.state_head(hidden)  # [B, K, 4]
        current_size = states[..., 2:4].exp()  # [B, K, 2]
        delta_center = (
            raw_delta[..., :2].tanh()
            * current_size
            * self.center_step
        )  # [B, K, 2]
        delta_size = raw_delta[..., 2:4].tanh() * self.size_step  # [B, K, 2]
        centers = (states[..., :2] + delta_center).clamp(0.0, 1.0)  # [B, K, 2]
        log_sizes = states[..., 2:4] + delta_size  # [B, K, 2]
        log_sizes = log_sizes.clamp(
            min=math.log(min_box_size), max=math.log(max_box_size)
        )  # [B, K, 2]
        new_states = torch.cat((centers, log_sizes), dim=-1)  # [B, K, 4]
        new_certainty = certainty_logits + self.certainty_head(hidden).squeeze(-1)  # [B, K]
        return new_states, new_certainty, correlations


class CrossViewReranker(nn.Module):
    """Rerank SVI candidates using prompt-local cross-view patch attention."""

    def __init__(
        self,
        query_channels: Dict[str, int],
        reference_channels: Dict[str, int],
        levels: Tuple[str, ...],
        hidden_dim: int,
        heads: int,
        layers: int,
        patch_size: int,
        query_patch_scale: float,
    ) -> None:
        """Build the multi-scale cross-attention candidate reranker."""
        super().__init__()
        if not levels:
            raise ValueError("CrossViewReranker requires at least one feature level")
        if hidden_dim % heads:
            raise ValueError("CrossViewReranker dimension must be divisible by heads")
        self.levels = tuple(levels)
        self.patch_size = patch_size
        self.query_patch_scale = max(query_patch_scale, 1e-4)
        # Per-view projections into the shared attention space.
        self.query_proj = nn.ModuleDict(
            {
                level: nn.Conv2d(query_channels[level], hidden_dim, 1)
                for level in self.levels
            }
        )
        self.reference_proj = nn.ModuleDict(
            {
                level: nn.Conv2d(reference_channels[level], hidden_dim, 1)
                for level in self.levels
            }
        )
        self.query_norm = nn.ModuleDict(
            {level: nn.LayerNorm(hidden_dim) for level in self.levels}
        )
        self.reference_norm = nn.ModuleDict(
            {level: nn.LayerNorm(hidden_dim) for level in self.levels}
        )
        layer_count = max(layers, 1)
        # Multi-scale query-to-reference cross-attention blocks.
        self.cross_attention = nn.ModuleDict(
            {
                level: nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            hidden_dim,
                            heads,
                            dropout=0.1,
                            batch_first=True,
                        )
                        for _ in range(layer_count)
                    ]
                )
                for level in self.levels
            }
        )
        self.cross_norm = nn.ModuleDict(
            {
                level: nn.ModuleList(
                    [nn.LayerNorm(hidden_dim) for _ in range(layer_count)]
                )
                for level in self.levels
            }
        )
        feature_dim = len(self.levels) * hidden_dim * 2 + 6
        # Residual candidate ranking head.
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        # Start as an exact residual no-op and let selection supervision learn the reranking.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    @staticmethod
    def _prompt_states(
        prompt_map: torch.Tensor,
        feature: torch.Tensor,
        patch_scale: float,
    ) -> torch.Tensor:
        """Convert a prompt map into one normalized query patch state per image."""
        if prompt_map.ndim == 3:
            prompt_map = prompt_map.unsqueeze(1)  # [B, 1, Hp, Wp]
        weights = F.interpolate(
            prompt_map.float(),
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0)  # [B, 1, Hf, Wf]
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)  # [B, 1, Hf, Wf]
        height, width = feature.shape[-2:]
        y_coords = (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) / height  # [Hf]
        x_coords = (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) / width  # [Wf]
        center_x = (weights[:, 0] * x_coords.view(1, 1, -1)).sum(dim=(-2, -1))  # [B]
        center_y = (weights[:, 0] * y_coords.view(1, -1, 1)).sum(dim=(-2, -1))  # [B]
        centers = torch.stack((center_x, center_y), dim=-1)  # [B, 2]
        sizes = centers.new_full((centers.shape[0], 2), patch_scale)  # [B, 2]
        return torch.cat((centers, sizes.log()), dim=-1).unsqueeze(1)  # [B, 1, 4]

    def forward(
        self,
        query_features: TensorDict,
        reference_features: TensorDict,
        prompt_map: torch.Tensor,
        states: torch.Tensor,
        anchor_scores: torch.Tensor,
        certainty_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return one residual ranking logit for each existing candidate."""
        query_states = self._prompt_states(
            prompt_map,
            query_features[self.levels[0]],
            self.query_patch_scale,
        )  # [B, 1, 4]
        detached_states = states.detach()  # [B, K, 4]
        batch, candidate_count = states.shape[:2]
        level_features = []
        for level in self.levels:
            query_feature = query_features[level].detach()  # [B, Cq, Hq, Wq]
            reference_feature = reference_features[level].detach()  # [B, Cr, Hr, Wr]
            query_patch = sample_candidate_patches(
                query_feature, query_states, self.patch_size
            )[:, 0]  # [B, Cq, S, S]
            reference_patches = sample_candidate_patches(
                reference_feature, detached_states, self.patch_size
            )  # [B, K, Cr, S, S]
            _, _, channels, patch_height, patch_width = reference_patches.shape
            query_tokens = self.query_proj[level](query_patch).flatten(2).transpose(1, 2)  # [B, S^2, D]
            reference_tokens = self.reference_proj[level](
                reference_patches.reshape(
                    batch * candidate_count,
                    channels,
                    patch_height,
                    patch_width,
                )
            ).flatten(2).transpose(1, 2)  # [B*K, S^2, D]
            query_tokens = self.query_norm[level](query_tokens)  # [B, S^2, D]
            reference_tokens = self.reference_norm[level](reference_tokens)  # [B*K, S^2, D]
            query_tokens = query_tokens[:, None].expand(
                -1, candidate_count, -1, -1
            ).reshape(batch * candidate_count, query_tokens.shape[1], -1)  # [B*K, S^2, D]
            attended = query_tokens  # [B*K, S^2, D]
            for attention, norm in zip(
                self.cross_attention[level], self.cross_norm[level]
            ):
                residual = attended  # [B*K, S^2, D]
                attended, _ = attention(
                    attended,
                    reference_tokens,
                    reference_tokens,
                    need_weights=False,
                )  # [B*K, S^2, D]
                attended = norm(attended + residual)  # [B*K, S^2, D]
            attended_pool = attended.mean(dim=1)  # [B*K, D]
            query_pool = query_tokens.mean(dim=1)  # [B*K, D]
            reference_pool = reference_tokens.mean(dim=1)  # [B*K, D]
            interaction = query_pool * reference_pool  # [B*K, D]
            level_features.append(
                torch.cat((attended_pool, interaction), dim=-1).view(
                    batch, candidate_count, -1
                )
            )  # Each element is [B, K, 2D]
        metadata = torch.cat(
            (
                detached_states,
                anchor_scores.detach().clamp_min(1e-8).log().unsqueeze(-1),
                certainty_logits.detach().unsqueeze(-1),
            ),
            dim=-1,
        )  # [B, K, 6]
        features = torch.cat((*level_features, metadata), dim=-1)  # [B, K, 2DL+6]
        return self.score_head(features).squeeze(-1)  # [B, K]





class PCWNet(nn.Module):
    """Probabilistic coarse-to-fine warp network for cross-view localization."""

    def __init__(self, config: Optional[PCWNetConfig] = None) -> None:
        """Assemble the coarse matcher, refiners, and optional SVI reranker."""
        super().__init__()
        self.config = config or PCWNetConfig()
        cfg = self.config

        # Shared coarse encoder and view-specific fine feature pyramids.
        self.coarse_encoder = CoarseSemanticEncoder(cfg.pretrained_backbones)
        self.query_fine_encoder = FineFeaturePyramid(
            cfg.pretrained_backbones
        )
        self.reference_fine_encoder = FineFeaturePyramid(
            cfg.pretrained_backbones
        )

        # Prompt-conditioned coarse anchor matching module.
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

        # Multi-scale candidate box refinement module.
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
        # Optional SVI-only cross-view candidate reranking module.
        self.reranker = None
        if cfg.reranker:
            self.reranker = CrossViewReranker(
                query_channels=FineFeaturePyramid.out_channels,
                reference_channels=FineFeaturePyramid.out_channels,
                levels=("layer3", "layer2"),
                hidden_dim=cfg.reranker_dim,
                heads=cfg.reranker_heads,
                layers=cfg.reranker_layers,
                patch_size=cfg.reranker_patch_size,
                query_patch_scale=cfg.query_patch_scale,
            )

        # Coarse-backbone freezing policy.
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

        # Coarse semantic encoding for global cross-view candidate search.
        query_coarse = self._encode_coarse(query_images)  # [B, 768, Hq/32, Wq/32]
        reference_coarse = self._encode_coarse(reference_images)  # [B, 768, Hr/32, Wr/32]
        # Fine feature pyramids for local object-level refinement.
        query_fine = self.query_fine_encoder(query_images)  # layer1/2/3: [B, 64/128/256, Hq/4/8/16, Wq/4/8/16]
        reference_fine = self.reference_fine_encoder(reference_images)  # layer1/2/3: [B, 64/128/256, Hr/4/8/16, Wr/4/8/16]

        # Prompt-conditioned coarse matching over the reference anchor grid.
        prompt_coarse = prompt_weighted_pool(query_coarse, prompt_map)  # [B, 768]
        prompt_token = self.query_coarse_proj(prompt_coarse)  # [B, D]
        reference_grid = F.adaptive_avg_pool2d(
            reference_coarse, cfg.anchor_grid_size
        )  # [B, 768, Ah, Aw]
        reference_grid = self.reference_coarse_proj(reference_grid)  # [B, D, Ah, Aw]
        batch, channels, anchor_h, anchor_w = reference_grid.shape  # B, D, Ah, Aw
        reference_tokens = reference_grid.flatten(2).transpose(1, 2)  # [B, Ah*Aw, D]

        # Anchor decoding and initial candidate construction.
        decoded = self.anchor_decoder(prompt_token, reference_tokens)  # anchor [B, N], size [B, N, 2]
        anchor_logits = decoded["anchor_logits_flat"].view(
            batch, anchor_h, anchor_w
        )  # [B, Ah, Aw]
        centers, anchor_indices, _ = local_soft_argmax_2d(
            anchor_logits,
            topk=cfg.topk,
            radius=cfg.local_softargmax_radius,
        )  # centers [B, K, 2], indices [B, K], scores [B, K]
        anchor_probabilities_flat = decoded["anchor_logits_flat"].softmax(dim=-1)  # [B, N]
        anchor_scores = torch.gather(
            anchor_probabilities_flat, 1, anchor_indices
        )  # [B, K]

        size_logits = torch.gather(
            decoded["size_logits_flat"],
            1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, 2),
        )  # [B, K, 2]
        initial_sizes = cfg.min_box_size + size_logits.sigmoid() * (
            cfg.max_box_size - cfg.min_box_size
        )  # [B, K, 2]
        states = torch.cat((centers, initial_sizes.log()), dim=-1)  # [B, K, 4]
        certainty_logits = torch.zeros_like(anchor_scores)  # [B, K]

        # Multi-scale object refinement over the candidate boxes.
        refinement_states: List[torch.Tensor] = [states]
        refinement_certainties: List[torch.Tensor] = [certainty_logits]
        local_correlations: List[torch.Tensor] = []
        for level in cfg.refinement_levels:
            query_token = prompt_weighted_pool(query_fine[level], prompt_map)  # [B, C_level]
            states, certainty_logits, correlations = self.refiners[level](
                query_token,
                reference_fine[level],
                states,
                certainty_logits,
                min_box_size=cfg.min_box_size,
                max_box_size=cfg.max_box_size,
            )  # states [B, K, 4], certainty [B, K], correlation [B, K, S, S]
            refinement_states.append(states)  # Each element [B, K, 4]
            refinement_certainties.append(certainty_logits)  # Each element [B, K]
            local_correlations.append(correlations)  # Each element [B, K, S, S]

        # Candidate score fusion, optional SVI reranking, and final selection.
        candidate_boxes = states_to_boxes(states)  # [B, K, 4]
        anchor_score_power = max(cfg.anchor_score_power, 0.0)  # scalar
        candidate_selection_logits = (
            anchor_score_power * anchor_scores.clamp_min(1e-8).log()
            + F.logsigmoid(certainty_logits)
        )  # [B, K]
        svi_rerank_logits = torch.zeros_like(candidate_selection_logits)  # [B, K]
        if self.reranker is not None:
            svi_rerank_logits = self.reranker(
                query_fine,
                reference_fine,
                prompt_map,
                states,
                anchor_scores,
                certainty_logits,
            )  # [B, K]
            candidate_selection_logits = candidate_selection_logits + (
                cfg.reranker_weight * svi_rerank_logits
            )  # [B, K]
        candidate_scores = candidate_selection_logits.exp()  # [B, K]
        selected_indices = candidate_scores.argmax(dim=1)  # [B]
        batch_indices = torch.arange(batch, device=query_images.device)  # [B]
        boxes = candidate_boxes[batch_indices, selected_indices]  # [B, 4]
        scores = candidate_scores[batch_indices, selected_indices]  # [B]

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
            "svi_rerank_logits": svi_rerank_logits,
            "candidate_scores": candidate_scores,
            "selected_indices": selected_indices,
            "boxes": boxes,
            "scores": scores,
        }
