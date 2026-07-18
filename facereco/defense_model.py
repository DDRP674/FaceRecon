from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MultiNoiseDefense(nn.Module):
    def __init__(self, embed_dim: int = 512, num_levels: int = 10, hidden_dim: int | None = None, dropout: float = 0.8):
        super().__init__()
        hidden_dim = hidden_dim or (384 if embed_dim == 512 else max(1, embed_dim * 3 // 4))
        self.level_logits = nn.Parameter(torch.zeros(num_levels))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, embeds_by_level: torch.Tensor) -> torch.Tensor:
        # embeds_by_level: [batch, num_levels, embed_dim]
        weights = torch.softmax(self.level_logits, dim=0)
        mixed = (embeds_by_level * weights[None, :, None]).sum(dim=1)
        return F.normalize(self.mlp(mixed), dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
        logits = cosine * (1.0 - one_hot) + target * one_hot
        return logits * self.scale

