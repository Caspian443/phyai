"""Numerical boundary alignment for RLinf's pi0.5 actor."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.pi05.configuration_pi05 import (
    PaliGemmaTextConfig,
    PI05Config,
)
from phyai.models.pi05.modeling_pi05 import (
    ActionTimeHeads,
    PI05Model,
    SiglipVisionEmbeddings,
)


class PI05RLVisionEmbeddings(SiglipVisionEmbeddings):
    """Match openpi_rlinf's FP32 stem with resident BF16 parameters."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be 4-D (B, C, H, W); got shape "
                f"{tuple(pixel_values.shape)}."
            )
        _, channels, height, width = pixel_values.shape
        expected = (
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if (channels, height, width) != expected:
            raise ValueError(
                f"pixel_values shape {tuple(pixel_values.shape)} does not match "
                f"config: expected (B, {expected[0]}, {expected[1]}, {expected[2]})."
            )

        patch = self.patch_embedding
        hidden = F.conv2d(
            pixel_values.float(),
            patch.weight.float(),
            patch.bias.float() if patch.bias is not None else None,
            patch.stride,
            patch.padding,
            patch.dilation,
            patch.groups,
        )
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = hidden + self.position_embedding().float()
        return hidden.to(patch.weight.dtype)


class PI05RLEmbedTokens(nn.Module):
    """Apply Gemma's scale as a Python scalar after the embedding lookup."""

    def __init__(
        self,
        config: PaliGemmaTextConfig,
        *,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.scale = math.sqrt(config.hidden_size)
        self.embedding = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * self.scale


class PI05RLActionTimeHeads(ActionTimeHeads):
    """Round timesteps to the actor compute dtype before SinCos."""

    def embed_time(self, time: torch.Tensor) -> torch.Tensor:
        return super().embed_time(time.to(self.time_mlp_in.weight.dtype))


class PI05RLModel(PI05Model):
    """PI05Model with only openpi_rlinf numerical boundaries replaced."""

    def __init__(self, config: PI05Config, **kwargs) -> None:
        if kwargs.get("vision_params_dtype") is not None:
            raise ValueError(
                "pi05_rl fixes resident vision parameters to the engine dtype; "
                "vision_params_dtype is only supported by the pi05 plugin."
            )
        super().__init__(config, **kwargs)

        vision_model = self.vision.vision_tower.vision_model
        vision_model.embeddings = PI05RLVisionEmbeddings(
            config.vision,
            params_dtype=self.params_dtype,
            prefix=vision_model.embeddings.prefix,
        )
        self.paligemma_lm.embed_tokens = PI05RLEmbedTokens(
            config.text,
            params_dtype=self.params_dtype,
            prefix="paligemma_with_expert.paligemma.lm_head",
        )
        self.heads = PI05RLActionTimeHeads(
            config,
            params_dtype=self.params_dtype,
        )


__all__ = [
    "PI05RLActionTimeHeads",
    "PI05RLEmbedTokens",
    "PI05RLModel",
    "PI05RLVisionEmbeddings",
]
