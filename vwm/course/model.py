"""Trained-from-scratch Siamese CNN for change detection.

Architecture (compact; ~1.5M params; trains in <10 min on a single GPU):

    Encoder  E(x): 4 BasicBlocks (Conv-BN-ReLU x2 + residual) with 3 strided poolings
                   -> spatial 300x400 -> 38x50 -> global avg pool -> 256-d embedding
    Compare  C([e_s, e_t, |e_s - e_t|, e_s * e_t]) -> MLP(512 -> 128 -> 1)
    Output   sigmoid(P(change))

The 4-way concatenation (sample, test, abs-diff, element-wise product) is the standard
contrastive comparison head used in Koch et al. 2015 (Siamese networks for one-shot
recognition); we adapt it to a 2-class change-detection problem.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if downsample or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = BasicBlock(32, 64, downsample=True)
        self.block2 = BasicBlock(64, 128, downsample=True)
        self.block3 = BasicBlock(128, 256, downsample=False)
        self.block4 = BasicBlock(256, embed_dim, downsample=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.pool(h).flatten(1)         # (B, embed_dim)
        return h


class SiameseCNN(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, sample: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
        e_s = self.encoder(sample)
        e_t = self.encoder(test)
        feats = torch.cat([e_s, e_t, (e_s - e_t).abs(), e_s * e_t], dim=1)
        logits = self.head(feats).squeeze(-1)
        return logits          # raw logits; use BCEWithLogitsLoss
