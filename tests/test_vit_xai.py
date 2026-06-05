"""Smoke tests for ViT detection + Attention Rollout.

Uses a tiny hand-rolled ViT-like module (one ``nn.MultiheadAttention``
block) so the test doesn't pull torchvision's full ViT weights or
require torch>=2.4. Verifies:

  * ``is_vision_transformer`` correctly tags the toy ViT and rejects
    plain CNNs (the toy_model fixture).
  * ``AttentionRolloutExplainer.compute`` returns a normalised
    ``[H, W]`` heatmap.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroquant.xai.explainability import (
    AttentionRolloutExplainer,
    is_vision_transformer,
)


class _TinyViT(nn.Module):
    """One-block toy ViT for testing.

    16x16 input, 4x4 patches (16 patches + 1 CLS = 17 tokens),
    embed_dim=8, 2 attention heads. Just enough surface area to
    exercise the rollout hooks without a real ViT's weight count.
    """

    def __init__(self, embed_dim: int = 8, num_heads: int = 2, img_size: int = 16, patch: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(patch * patch * 3, embed_dim)
        self.patch = patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Single attention block — enough to verify the rollout multiplies
        # at least one matrix and unwraps the CLS row.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch
        # Unfold into patches: [B, num_patches, patch*patch*3]
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, c, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, c * p * p)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x, _ = self.attn(x, x, x)
        x = self.norm(x)
        return self.head(x[:, 0])


def test_is_vision_transformer_detects_vit() -> None:
    assert is_vision_transformer(_TinyViT()) is True


def test_is_vision_transformer_rejects_cnn(toy_model) -> None:
    # toy_model fixture is the CNN from tests/conftest.py
    assert is_vision_transformer(toy_model) is False


def test_attention_rollout_returns_image_sized_heatmap() -> None:
    model = _TinyViT().eval()
    explainer = AttentionRolloutExplainer(model, device=torch.device("cpu"))
    img = torch.randn(1, 3, 16, 16)
    heatmap = explainer.compute(img)
    # Heatmap should match the input spatial dims (the rollout upsamples
    # internally).
    assert heatmap.shape == (16, 16)
    assert heatmap.dtype == np.float64 or heatmap.dtype == np.float32
    # Normalised into [0, 1].
    assert heatmap.min() >= 0.0 - 1e-6
    assert heatmap.max() <= 1.0 + 1e-6
