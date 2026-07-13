from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EEGNetOutput:
    logits: torch.Tensor
    embedding: torch.Tensor


class EEGNet(nn.Module):
    """Compact EEGNet-style network for 19-channel four-second epochs.

    Input shape is ``(batch, channels, samples)``. The classifier returns both
    logits and the learned embedding so the fusion model can be ablated without
    duplicating the raw-signal branch.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_classes: int,
        temporal_kernel: int = 125,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        dropout: float = 0.5,
        embedding_bins: int = 8,
    ) -> None:
        super().__init__()
        padding = temporal_kernel // 2
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, padding), bias=False),
            nn.BatchNorm2d(f1),
        )
        spatial_channels = f1 * depth_multiplier
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                spatial_channels,
                (n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                (1, 16),
                padding=(0, 8),
                groups=spatial_channels,
                bias=False,
            ),
            nn.Conv2d(spatial_channels, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, embedding_bins)),
        )
        self.embedding_size = f2 * embedding_bins
        self.classifier = nn.Linear(self.embedding_size, n_classes)

    def forward(self, values: torch.Tensor) -> EEGNetOutput:
        if values.ndim != 3:
            raise ValueError("EEGNet input must have shape (batch, channels, samples)")
        hidden = self.temporal(values.unsqueeze(1))
        hidden = self.spatial(hidden)
        hidden = self.separable(hidden)
        embedding = hidden.flatten(start_dim=1)
        return EEGNetOutput(logits=self.classifier(embedding), embedding=embedding)


class ComplexityFusionNet(nn.Module):
    """Late fusion of EEGNet embeddings and fold-standardized biomarkers."""

    def __init__(
        self,
        eegnet: EEGNet,
        *,
        n_features: int,
        n_classes: int,
        feature_hidden: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.eegnet = eegnet
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, feature_hidden),
            nn.LayerNorm(feature_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(eegnet.embedding_size + feature_hidden, n_classes)

    def forward(self, eeg: torch.Tensor, features: torch.Tensor) -> EEGNetOutput:
        raw_output = self.eegnet(eeg)
        feature_embedding = self.feature_encoder(features)
        joint = torch.cat([raw_output.embedding, feature_embedding], dim=1)
        return EEGNetOutput(logits=self.classifier(joint), embedding=joint)

