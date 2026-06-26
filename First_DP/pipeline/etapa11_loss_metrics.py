#!/usr/bin/env python3
"""
Etapa 11 — Função de perda, otimizador e métricas
=================================================
Define loss, Adam e métricas de regressão para amplitude do Vicon.

NÃO executa loop de treino completo. Apenas configura e demonstra cálculos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from etapa01_setup import ExperimentConfig, build_default_config, get_device
from curve_utils import curve_window_mae, curve_window_rms, curve_window_rmse, is_curve_array
from etapa06_normalize import ScalerBundle
from etapa08_dataloader import FoldDataLoaders, run_stage08_dataloader
from etapa10_tcn import TCNRegressor, build_tcn_from_config

LossName = Literal["mse", "mae", "huber"]


# ---------------------------------------------------------------------------
# 1. Configuração de treino
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Hiperparâmetros de otimização (usados nas Etapas 12–15).

    loss           : função usada para backprop (gradiente)
    learning_rate  : passo do Adam
    weight_decay   : regularização L2 nos pesos
    huber_delta    : transição quadrática→linear na HuberLoss
    """

    loss: LossName = "huber"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    huber_delta: float = 1.0
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4
    dropout: float = 0.2
    num_workers: int = 0
    pin_memory: bool = False
    use_amp: bool = False


# ---------------------------------------------------------------------------
# 2. Função de perda
# ---------------------------------------------------------------------------
def build_criterion(train_cfg: TrainingConfig) -> nn.Module:
    """
    Cria a loss usada no treino (backprop).

    MSE   : penaliza erros grandes quadraticamente — sensível a outliers
    MAE   : robusta, mas não suave em zero (gradiente descontínuo)
    Huber : híbrida — quadrática para erros pequenos, linear para grandes
    """
    if train_cfg.loss == "mse":
        return nn.MSELoss()
    if train_cfg.loss == "mae":
        return nn.L1Loss()
    if train_cfg.loss == "huber":
        return nn.HuberLoss(delta=train_cfg.huber_delta)
    raise ValueError(f"Loss desconhecida: {train_cfg.loss!r}")


def compute_batch_loss(
    criterion: nn.Module,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Loss escalar para um mini-batch (usada em loss.backward())."""
    return criterion(y_pred, y_true)


# ---------------------------------------------------------------------------
# 3. Otimizador
# ---------------------------------------------------------------------------
def build_optimizer(
    model: nn.Module,
    train_cfg: TrainingConfig,
) -> torch.optim.Adam:
    """
    Adam: taxa de aprendizado adaptativa por parâmetro.

    weight_decay aplica regularização L2 — ajuda a evitar overfitting
    quando a TCN tem mais parâmetros que a CNN 1D.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )


# ---------------------------------------------------------------------------
# 4. Métricas de interpretação (não necessariamente = loss de treino)
# ---------------------------------------------------------------------------
@dataclass
class RegressionMetrics:
    """
    Métricas para interpretar desempenho clínico/estatístico.

    Diferente da loss de treino: aqui queremos unidades e significado.
    """

    mae: float
    rmse: float
    rms: float
    r2: float
    bias: float
    mape_pct: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_regression_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    epsilon: float = 1e-8,
) -> RegressionMetrics:
    """
    Métricas de regressão.

    Escalar (amplitude): arrays 1D.
    Curva: arrays (n_janelas, T) — MAE/RMSE/RMS são médias por janela.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.shape != yp.shape:
        raise ValueError(f"Shapes diferentes: y_true={yt.shape}, y_pred={yp.shape}")
    if yt.size == 0:
        raise ValueError("Arrays vazios.")

    if is_curve_array(yt):
        mae = curve_window_mae(yt, yp)
        rmse = curve_window_rmse(yt, yp)
        rms = curve_window_rms(yt, yp)
        yt_flat = yt.ravel()
        yp_flat = yp.ravel()
    else:
        yt_flat = yt.ravel()
        yp_flat = yp.ravel()
        mae = float(mean_absolute_error(yt_flat, yp_flat))
        rmse = float(np.sqrt(mean_squared_error(yt_flat, yp_flat)))
        rms = rmse

    r2 = float(r2_score(yt_flat, yp_flat)) if len(yt_flat) > 1 else float("nan")
    bias = float(np.mean(yp_flat - yt_flat))
    mape = float(np.mean(np.abs(yp_flat - yt_flat) / (np.abs(yt_flat) + epsilon)) * 100.0)

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        rms=rms,
        r2=r2,
        bias=bias,
        mape_pct=mape,
        n_samples=int(yt_flat.size),
    )


def metrics_to_original_scale(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler_bundle: ScalerBundle,
) -> tuple[np.ndarray, np.ndarray]:
    """Reverte predições/alvos para cm, preservando shape (incl. curvas 2D)."""
    shape = np.asarray(y_true_scaled).shape
    yt = scaler_bundle.inverse_transform_target(y_true_scaled).reshape(shape)
    yp = scaler_bundle.inverse_transform_target(y_pred_scaled).reshape(shape)
    return yt, yp


# ---------------------------------------------------------------------------
# 5. Avaliação de um loader (sem treinar)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, RegressionMetrics, np.ndarray, np.ndarray]:
    """
    Percorre um DataLoader e agrega loss média + métricas.

    Retorna: (loss_média, métricas, y_true, y_pred) na escala atual dos dados.
    """
    model.eval()
    losses: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = model(x)

        loss = compute_batch_loss(criterion, y_hat, y)
        losses.append(float(loss.item()))

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    metrics = compute_regression_metrics(y_true, y_pred)
    mean_loss = float(np.mean(losses)) if losses else float("nan")

    return mean_loss, metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# 6. Exemplo mínimo ilustrativo (arrays fictícios)
# ---------------------------------------------------------------------------
def make_illustrative_example() -> tuple[np.ndarray, np.ndarray]:
    """
    Pequeno exemplo numérico para ensinar métricas — não são dados reais.
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    y_pred = np.array([1.2, 1.8, 3.5, 3.7, 5.4], dtype=float)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# 7. Relatório legível
# ---------------------------------------------------------------------------
def print_loss_metrics_report(
    train_cfg: TrainingConfig,
    criterion: nn.Module,
    example_metrics: RegressionMetrics,
    val_loss: float,
    val_metrics: RegressionMetrics,
    train_loss: float,
    train_metrics: RegressionMetrics,
) -> None:
    print("=" * 60)
    print("ETAPA 11 — Função de perda, otimizador e métricas")
    print("=" * 60)
    print(f"Loss de treino      : {train_cfg.loss.upper()}", end="")
    if train_cfg.loss == "huber":
        print(f" (delta={train_cfg.huber_delta})")
    else:
        print()
    print(f"Learning rate       : {train_cfg.learning_rate}")
    print(f"Weight decay        : {train_cfg.weight_decay}")
    print(f"Batch size          : {train_cfg.batch_size}")
    print(f"Max epochs          : {train_cfg.max_epochs}  (patience={train_cfg.patience})")

    print("\n--- Exemplo ilustrativo (5 pontos fictícios) ---")
    print(f"  MAE   : {example_metrics.mae:.4f}")
    print(f"  RMSE  : {example_metrics.rmse:.4f}")
    print(f"  R²    : {example_metrics.r2:.4f}")
    print(f"  Bias  : {example_metrics.bias:+.4f}")
    print(f"  MAPE  : {example_metrics.mape_pct:.2f}%")

    print("\n--- Modelo NÃO treinado — fold exemplo (escala vicon normalizada) ---")
    print(f"  Treino — loss ({train_cfg.loss}): {train_loss:.4f}")
    print(f"  Treino — MAE (métrica)           : {train_metrics.mae:.4f}")
    print(f"  Val    — loss ({train_cfg.loss}): {val_loss:.4f}")
    print(f"  Val    — MAE (métrica)           : {val_metrics.mae:.4f}")
    print(f"  Val    — RMSE                   : {val_metrics.rmse:.4f}")
    print(f"  Val    — R²                     : {val_metrics.r2:.4f}")
    print(f"  Val    — Bias                   : {val_metrics.bias:+.4f}")

    print("\nLoss de treino vs métrica de interpretação:")
    print("  - Loss (MSE/Huber): o que o modelo otimiza via gradiente.")
    print("  - MAE/RMSE/R²    : o que você reporta no artigo/tese.")
    print("  - Se vicon foi escalado: use inverse_transform para MAE em cm.")

    print("\nPor que Huber é boa candidata aqui:")
    print("  - robusta a janelas com picos/outliers de amplitude")
    print("  - mais estável que MSE pura, mais suave que MAE pura")

    print("\nPróxima etapa: treino de um único fold (Etapa 12).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 8. Ponto de entrada da Etapa 11
# ---------------------------------------------------------------------------
def run_stage11_loss_metrics(
    config: ExperimentConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    loaders: FoldDataLoaders | None = None,
) -> tuple[TrainingConfig, nn.Module, torch.optim.Adam, RegressionMetrics]:
    """
    Configura loss/optimizer e demonstra métricas.

    Avalia modelo TCN inicializado (não treinado) em um fold de exemplo.
    """
    if config is None:
        config = build_default_config()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    device = get_device()

    if loaders is None:
        loaders = run_stage08_dataloader(config=config, batch_size=train_cfg.batch_size)

    model = build_tcn_from_config(config).to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)

    # exemplo didático com arrays fictícios
    yt_ex, yp_ex = make_illustrative_example()
    example_metrics = compute_regression_metrics(yt_ex, yp_ex)

    train_loss, train_metrics, _, _ = evaluate_loader(
        model, loaders.train, criterion, device
    )
    val_loss, val_metrics, _, _ = evaluate_loader(
        model, loaders.val, criterion, device
    )

    print_loss_metrics_report(
        train_cfg,
        criterion,
        example_metrics,
        val_loss,
        val_metrics,
        train_loss,
        train_metrics,
    )

    return train_cfg, criterion, optimizer, val_metrics


if __name__ == "__main__":
    run_stage11_loss_metrics()
