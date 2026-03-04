"""
인코더 모듈: TextEncoder, PosteriorEncoder.
"""
from __future__ import annotations

import math
import sys
import os

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from torch import nn

import commons  # type: ignore[import]
import modules  # type: ignore[import]
import attentions  # type: ignore[import]


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.n_vocab          = n_vocab
        self.out_channels     = out_channels
        self.hidden_channels  = hidden_channels
        self.filter_channels  = filter_channels
        self.n_heads          = n_heads
        self.n_layers         = n_layers
        self.kernel_size      = kernel_size
        self.p_dropout        = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)                       # [b, h, t]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size     = kernel_size
        self.dilation_rate   = dilation_rate
        self.n_layers        = n_layers
        self.gin_channels    = gin_channels

        self.pre  = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc  = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
