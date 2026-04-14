"""Metric-learning and baseline classification losses for face recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss implemented from scratch."""

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos_theta = (embeddings @ W_norm.T).clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        theta_m = theta + one_hot * self.margin
        logits = self.scale * torch.cos(theta_m)
        return F.cross_entropy(logits, labels)


class CosFaceLoss(nn.Module):
    """CosFace large margin cosine loss implemented from scratch."""

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        scale: float = 64.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.W = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        W_norm = F.normalize(self.W, p=2, dim=1)
        cos_theta = (embeddings @ W_norm.T).clamp(-1 + 1e-7, 1 - 1e-7)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = self.scale * (cos_theta - one_hot * self.margin)
        return F.cross_entropy(logits, labels)


class SoftmaxHead(nn.Module):
    """Linear softmax baseline head over normalized embeddings."""

    def __init__(self, embedding_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.fc(embeddings), labels)
