"""Model definitions for face embedding extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):
    """MobileNetV2 backbone + projection head for L2-normalized embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        base = models.mobilenet_v2(pretrained=pretrained)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)
