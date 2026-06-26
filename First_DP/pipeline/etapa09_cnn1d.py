#!/usr/bin/env python3
"""
Etapa 9 — Modelo inicial: CNN 1D
================================
Rede convolucional 1D simples para regressão de amplitude do Vicon
a partir de janelas IMU (6 canais × T amostras).

NÃO treina o pipeline completo. Apenas define arquitetura e forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from etapa01_setup import ExperimentConfig, build_default_config, get_device
from etapa08_dataloader import FoldDataLoaders, run_stage08_dataloader


# ---------------------------------------------------------------------------
# 1. Bloco convolucional reutilizável
# ---------------------------------------------------------------------------
class ConvBlock1D(nn.Module):
    """
    Bloco Conv1d + BatchNorm + ReLU + MaxPool.

    Conv1d
    ------
    Desliza um filtro ao longo do **tempo** para cada canal de entrada.
    Parâmetros principais:
      - in_channels  : canais de entrada (ex.: 6 IMU)
      - out_channels : filtros aprendidos (features temporais)
      - kernel_size  : largura da janela local no tempo
      - padding      : mantém o tamanho temporal (com padding='same' implícito)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 2. CNN 1D para regressão escalar
# ---------------------------------------------------------------------------
class CNN1DRegressor(nn.Module):
    """
    CNN 1D: IMU (6 × T) -> amplitude escalar ou curva Vicon (T pontos).

    target_mode amplitude : saída (batch,)
    target_mode curve     : saída (batch, T) via upsample + head conv
    """

    def __init__(
        self,
        n_input_channels: int = 6,
        window_size: int = 120,
        conv_channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
        head_hidden: int = 64,
        dropout: float = 0.2,
        target_mode: str = "curve",
    ) -> None:
        super().__init__()
        self.n_input_channels = n_input_channels
        self.window_size = window_size
        self.target_mode = target_mode

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels e kernel_sizes devem ter o mesmo comprimento.")

        blocks: list[nn.Module] = []
        in_ch = n_input_channels
        for out_ch, k in zip(conv_channels, kernel_sizes):
            blocks.append(ConvBlock1D(in_ch, out_ch, kernel_size=k, pool_size=2, dropout=dropout))
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if target_mode == "curve":
            self.head = nn.Conv1d(conv_channels[-1], 1, kernel_size=1)
            self.scalar_head = None
        else:
            self.head = None
            self.scalar_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(conv_channels[-1], head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        if self.target_mode == "curve":
            x = torch.nn.functional.interpolate(
                x, size=self.window_size, mode="linear", align_corners=False
            )
            return self.head(x).squeeze(1)
        x = self.global_pool(x)
        return self.scalar_head(x).squeeze(-1)

    def forward_with_shapes(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...]]]]:
        """Forward pass com rastreamento de shapes (didático)."""
        shapes: list[tuple[str, tuple[int, ...]]] = [("entrada", tuple(x.shape))]

        h = x
        for i, block in enumerate(self.conv_layers):
            h = block(h)
            shapes.append((f"conv_block_{i + 1}", tuple(h.shape)))

        h = self.global_pool(h)
        shapes.append(("global_avg_pool", tuple(h.shape)))

        h = self.head(h)
        shapes.append(("saída (após head)", tuple(h.shape)))

        return h.squeeze(-1), shapes


# ---------------------------------------------------------------------------
# 3. Utilitários
# ---------------------------------------------------------------------------
@dataclass
class ModelSummary:
    name: str
    n_parameters: int
    n_trainable: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_cnn1d_from_config(config: ExperimentConfig, dropout: float | None = None) -> CNN1DRegressor:
    return CNN1DRegressor(
        n_input_channels=config.n_input_channels,
        window_size=config.window_samples,
        dropout=dropout if dropout is not None else 0.2,
        target_mode=config.target_mode,
    )


def summarize_forward(
    model: CNN1DRegressor,
    x: torch.Tensor,
) -> tuple[ModelSummary, list[tuple[str, tuple[int, ...]]], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        y, shapes = model.forward_with_shapes(x)

    total, trainable = count_parameters(model)
    summary = ModelSummary(
        name="CNN1DRegressor",
        n_parameters=total,
        n_trainable=trainable,
        input_shape=tuple(x.shape),
        output_shape=tuple(y.shape),
    )
    return summary, shapes, y


# ---------------------------------------------------------------------------
# 4. Relatório legível
# ---------------------------------------------------------------------------
def print_cnn1d_report(
    config: ExperimentConfig,
    model: CNN1DRegressor,
    summary: ModelSummary,
    shapes: list[tuple[str, tuple[int, ...]]],
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    device: torch.device,
) -> None:
    print("=" * 60)
    print("ETAPA 9 — Modelo inicial: CNN 1D")
    print("=" * 60)
    print(f"Dispositivo         : {device}")
    print(f"Entrada esperada    : (batch, {config.n_input_channels}, {config.window_samples})")
    print(f"Saída               : (batch,) — amplitude escalar")
    print(f"Parâmetros totais   : {summary.n_parameters:,}")
    print(f"Parâmetros treináveis: {summary.n_trainable:,}")

    print("\nForward pass — evolução das shapes:")
    for name, shape in shapes:
        print(f"  {name:<20} {shape}")

    print("\nExemplo com mini-batch real:")
    print(f"  y_true (3 primeiros): {y_true[:3].tolist()}")
    print(f"  y_pred (3 primeiros): {y_pred[:3].tolist()}")
    print("  (predições aleatórias — modelo ainda NÃO treinado)")

    print("\nPor que CNN 1D é um bom primeiro modelo para IMU:")
    print("  - Captura padrões locais no tempo (impulsos, oscilações).")
    print("  - Compartilha pesos ao longo da janela (eficiente).")
    print("  - Trata os 6 canais como 'mapa de features' multivariado.")
    print("  - Mais simples que TCN/LSTM — baseline forte para comparar.")

    print("\nComponentes principais:")
    print("  Conv1d      : filtros temporais aprendidos")
    print("  BatchNorm   : estabiliza treino")
    print("  MaxPool1d   : reduz dimensão temporal")
    print("  AdaptiveAvgPool : resume a janela inteira")
    print("  Linear      : regressão escalar final")

    print("\nPróxima etapa: TCN (Etapa 10).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 5. Ponto de entrada da Etapa 9
# ---------------------------------------------------------------------------
def run_stage09_cnn1d(
    config: ExperimentConfig | None = None,
    loaders: FoldDataLoaders | None = None,
) -> tuple[CNN1DRegressor, ModelSummary]:
    """
    Instancia CNN 1D e executa forward pass com um mini-batch real.

    NÃO treina. NÃO avalia no teste final.
    """
    if config is None:
        config = build_default_config()

    device = get_device()

    if loaders is None:
        loaders = run_stage08_dataloader(config=config)

    batch = next(iter(loaders.train))
    x = batch["x"].to(device)
    y_true = batch["y"].to(device)

    model = build_cnn1d_from_config(config).to(device)
    summary, shapes, y_pred = summarize_forward(model, x)

    print_cnn1d_report(config, model, summary, shapes, y_pred, y_true, device)
    return model, summary


if __name__ == "__main__":
    run_stage09_cnn1d()
