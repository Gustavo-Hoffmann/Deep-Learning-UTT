#!/usr/bin/env python3
"""TCN residual para predição ponto a ponto de resíduo temporal."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


def _make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int,
    dilation: int,
    *,
    dml_safe: bool,
) -> nn.Conv1d:
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    if dml_safe:
        return conv
    return weight_norm(conv)


class TemporalBlock(nn.Module):
    """Bloco convolucional 1D causal com conexão residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.15,
        *,
        dml_safe: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = _make_conv(in_channels, out_channels, kernel_size, padding, dilation, dml_safe=dml_safe)
        self.chomp1 = Chomp1d(padding)
        if dml_safe:
            self.norm1 = nn.Identity()
            self.act1 = nn.ReLU(inplace=True)
        else:
            self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
            self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _make_conv(out_channels, out_channels, kernel_size, padding, dilation, dml_safe=dml_safe)
        self.chomp2 = Chomp1d(padding)
        if dml_safe:
            self.norm2 = nn.Identity()
            self.act2 = nn.ReLU(inplace=True)
        else:
            self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
            self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.dml_safe = dml_safe
        self.out_act = nn.ReLU(inplace=True) if dml_safe else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.act1(self.norm1(self.chomp1(self.conv1(x)))))
        out = self.drop2(self.act2(self.norm2(self.chomp2(self.conv2(out)))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class ResidualTCN(nn.Module):
    """
    TCN que prediz resíduo temporal (batch, T) a partir de (batch, C, T).

    Entrada padrão: 10 canais.
    Saída: 1 resíduo por timestep (cm, possivelmente normalizado).

    dml_safe=True desativa GroupNorm/weight_norm/GELU (compatível com DirectML).
    """

    def __init__(
        self,
        n_input_channels: int = 10,
        window_size: int = 512,
        num_channels: int = 64,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        dropout: float = 0.15,
        *,
        dml_safe: bool = False,
    ) -> None:
        super().__init__()
        self.n_input_channels = n_input_channels
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dml_safe = dml_safe

        layers: list[nn.Module] = []
        in_ch = n_input_channels
        for d in dilations:
            layers.append(
                TemporalBlock(
                    in_ch,
                    num_channels,
                    kernel_size,
                    d,
                    dropout=dropout,
                    dml_safe=dml_safe,
                )
            )
            in_ch = num_channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(num_channels, 1, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)
        if h.shape[-1] != x.shape[-1]:
            h = nn.functional.interpolate(h, size=x.shape[-1], mode="linear", align_corners=False)
        return self.head(h).squeeze(1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
