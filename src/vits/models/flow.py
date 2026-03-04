"""
Normalizing Flow: ResidualCouplingBlock.
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from torch import nn
import modules  # type: ignore[import]


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.channels         = channels
        self.hidden_channels  = hidden_channels
        self.kernel_size      = kernel_size
        self.dilation_rate    = dilation_rate
        self.n_layers         = n_layers
        self.n_flows          = n_flows
        self.gin_channels     = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size,
                    dilation_rate, n_layers,
                    gin_channels=gin_channels, mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
