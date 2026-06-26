#!/usr/bin/env python3
"""
Etapa 10 — Modelo principal: TCN
==================================
Temporal Convolutional Network com convoluções dilatadas e conexões
residuais para regressão de amplitude do Vicon a partir de janelas IMU.

NÃO treina o pipeline completo. Apenas define arquitetura e forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from etapa01_setup import ExperimentConfig, build_default_config, get_device
from etapa08_dataloader import FoldDataLoaders, run_stage08_dataloader
from etapa09_cnn1d import CNN1DRegressor, ModelSummary, build_cnn1d_from_config, count_parameters


# ---------------------------------------------------------------------------
# 1. Bloco temporal dilatado (núcleo da TCN)
# ---------------------------------------------------------------------------
class Chomp1d(nn.Module):
    """
    Remove padding causal à direita para manter alinhamento temporal.

    Em TCN causal, convoluções adicionam padding à esquerda; Chomp1d
    corta o excesso à direita para que o output no tempo t dependa
    apenas de entradas até t.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    """
    Dois Conv1d dilatados + residual.

    Dilatação
    ---------
    dilation=1 : vizinhos imediatos
    dilation=2 : salta 1 amostra entre pesos do filtro
    dilation=4 : salta 3 amostras — campo receptivo cresce exponencialmente

    Com kernel=3 e dilatações [1,2,4,8], cada bloco enxerga mais
    contexto temporal sem aumentar o número de parâmetros.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + residual)


# ---------------------------------------------------------------------------
# 2. TCN para regressão escalar
# ---------------------------------------------------------------------------
class TCNRegressor(nn.Module):
    """
    TCN: IMU (6 × T) -> amplitude escalar ou curva Vicon (T pontos).

    target_mode amplitude : (batch,)
    target_mode curve     : (batch, T) — mantém resolução temporal
    """

    def __init__(
        self,
        n_input_channels: int = 6,
        window_size: int = 120,
        num_channels: tuple[int, ...] = (32, 32, 64, 64, 64),
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
        head_hidden: int = 64,
        target_mode: str = "curve",
    ) -> None:
        super().__init__()
        self.n_input_channels = n_input_channels
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.target_mode = target_mode

        if len(num_channels) != len(dilations):
            raise ValueError("num_channels e dilations devem ter o mesmo comprimento.")

        layers: list[nn.Module] = []
        in_ch = n_input_channels
        for out_ch, d in zip(num_channels, dilations):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, d, dropout=dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if target_mode == "curve":
            self.curve_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
            self.scalar_head = None
        else:
            self.curve_head = None
            self.scalar_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_channels[-1], head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

    @property
    def receptive_field(self) -> int:
        """
        Campo receptivo efetivo (causal) em número de amostras.

        Fórmula: 1 + Σ 2 × (kernel_size - 1) × dilation  (2 convs por bloco)
        """
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        if self.target_mode == "curve":
            if x.shape[-1] != self.window_size:
                x = torch.nn.functional.interpolate(
                    x, size=self.window_size, mode="linear", align_corners=False
                )
            return self.curve_head(x).squeeze(1)
        x = self.global_pool(x)
        return self.scalar_head(x).squeeze(-1)

    def forward_with_shapes(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...]]]]:
        shapes: list[tuple[str, tuple[int, ...]]] = [("entrada", tuple(x.shape))]

        h = x
        for i, block in enumerate(self.tcn):
            h = block(h)
            d = self.dilations[i]
            shapes.append((f"temporal_block_d={d}", tuple(h.shape)))

        h = self.global_pool(h)
        shapes.append(("global_avg_pool", tuple(h.shape)))

        h = self.head(h)
        shapes.append(("saída (após head)", tuple(h.shape)))

        return h.squeeze(-1), shapes


# ---------------------------------------------------------------------------
# 3. Utilitários e comparação CNN 1D vs TCN
# ---------------------------------------------------------------------------
def build_tcn_from_config(config: ExperimentConfig, dropout: float | None = None) -> TCNRegressor:
    return TCNRegressor(
        n_input_channels=config.n_input_channels,
        window_size=config.window_samples,
        dropout=dropout if dropout is not None else 0.2,
        target_mode=config.target_mode,
    )


def summarize_tcn_forward(
    model: TCNRegressor,
    x: torch.Tensor,
) -> tuple[ModelSummary, list[tuple[str, tuple[int, ...]]], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        y, shapes = model.forward_with_shapes(x)

    total, trainable = count_parameters(model)
    summary = ModelSummary(
        name="TCNRegressor",
        n_parameters=total,
        n_trainable=trainable,
        input_shape=tuple(x.shape),
        output_shape=tuple(y.shape),
    )
    return summary, shapes, y


@dataclass
class ModelComparison:
    cnn_params: int
    tcn_params: int
    cnn_receptive_field_approx: str
    tcn_receptive_field: int
    window_size: int
    tcn_covers_full_window: bool


def compare_cnn_vs_tcn(config: ExperimentConfig) -> ModelComparison:
    cnn = build_cnn1d_from_config(config)
    tcn = build_tcn_from_config(config)

    cnn_params, _ = count_parameters(cnn)
    tcn_params, _ = count_parameters(tcn)

    # CNN 1D com 3 pools: campo local cresce, mas pooling degrada resolução
    # Aproximação didática do alcance efetivo após 3 blocos (k=7,5,3 + pool)
    cnn_rf_note = "~15–30 amostras finais agregadas via pool (contexto local)"

    rf = tcn.receptive_field
    return ModelComparison(
        cnn_params=cnn_params,
        tcn_params=tcn_params,
        cnn_receptive_field_approx=cnn_rf_note,
        tcn_receptive_field=rf,
        window_size=config.window_samples,
        tcn_covers_full_window=rf >= config.window_samples,
    )


# ---------------------------------------------------------------------------
# 4. Relatório legível
# ---------------------------------------------------------------------------
def print_tcn_report(
    config: ExperimentConfig,
    model: TCNRegressor,
    summary: ModelSummary,
    shapes: list[tuple[str, tuple[int, ...]]],
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    comparison: ModelComparison,
    device: torch.device,
) -> None:
    print("=" * 60)
    print("ETAPA 10 — Modelo principal: TCN")
    print("=" * 60)
    print(f"Dispositivo          : {device}")
    print(f"Entrada esperada     : (batch, {config.n_input_channels}, {config.window_samples})")
    print(f"Saída                : (batch,) — amplitude escalar")
    print(f"Parâmetros totais    : {summary.n_parameters:,}")
    print(f"Campo receptivo TCN  : {model.receptive_field} amostras "
          f"({model.receptive_field / config.sampling_hz:.2f} s a {config.sampling_hz:.0f} Hz)")
    print(f"Cobre janela inteira : {'sim' if comparison.tcn_covers_full_window else 'não'}")

    print("\nDilatações por bloco:", list(model.dilations))
    print("\nForward pass — evolução das shapes:")
    for name, shape in shapes:
        print(f"  {name:<22} {shape}")

    print("\nExemplo com mini-batch real:")
    print(f"  y_true (3 primeiros): {y_true[:3].tolist()}")
    print(f"  y_pred (3 primeiros): {y_pred[:3].tolist()}")
    print("  (predições aleatórias — modelo ainda NÃO treinado)")

    print("\nComparação CNN 1D vs TCN:")
    print(f"  CNN 1D — parâmetros : {comparison.cnn_params:,}")
    print(f"  TCN    — parâmetros : {comparison.tcn_params:,}")
    print(f"  CNN 1D — contexto   : {comparison.cnn_receptive_field_approx}")
    print(f"  TCN    — contexto   : {comparison.tcn_receptive_field} amostras (campo receptivo causal)")

    print("\nQuando a TCN tende a ser melhor que CNN 1D:")
    print("  - padrões que dependem de contexto longo na janela (ex.: 1–2 s)")
    print("  - relação entre fases distantes do movimento")
    print("  - quando pooling da CNN perde informação temporal útil")

    print("\nQuando CNN 1D pode bastar:")
    print("  - padrões muito locais (picos curtos de aceleração)")
    print("  - poucos dados / risco de overfitting com TCN maior")
    print("  - baseline mais rápida de treinar e interpretar")

    print("\nComponentes TCN:")
    print("  Conv1d dilatada : expande alcance temporal sem pooling")
    print("  Chomp1d         : mantém causalidade (sem olhar o futuro)")
    print("  Residual        : estabiliza blocos empilhados")
    print("  weight_norm     : regulariza filtros convolucionais")

    print("\nPróxima etapa: loss, optimizer e métricas (Etapa 11).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 5. Ponto de entrada da Etapa 10
# ---------------------------------------------------------------------------
def run_stage10_tcn(
    config: ExperimentConfig | None = None,
    loaders: FoldDataLoaders | None = None,
) -> tuple[TCNRegressor, ModelSummary, ModelComparison]:
    """
    Instancia TCN e executa forward pass com mini-batch real.

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

    model = build_tcn_from_config(config).to(device)
    summary, shapes, y_pred = summarize_tcn_forward(model, x)
    comparison = compare_cnn_vs_tcn(config)

    print_tcn_report(config, model, summary, shapes, y_pred, y_true, comparison, device)
    return model, summary, comparison


if __name__ == "__main__":
    run_stage10_tcn()
