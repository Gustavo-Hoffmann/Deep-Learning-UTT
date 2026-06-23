#!/usr/bin/env python3
"""
Etapa 12 — Treinamento de um único fold
=======================================
Loop de treino + validação com early stopping para um fold LOSO
dentro dos sujeitos de desenvolvimento (70%).

NÃO executa LOSO completo (Etapa 13). NÃO usa teste final (30%).
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs, get_device, set_seed
from etapa03_load import LoadedDataset, load_all_subjects
from etapa05_split import apply_split_to_dataset, load_subject_split
from etapa06_normalize import FoldNormalizationResult, normalize_loso_fold
from etapa07_windows import create_windows_from_dataset
from etapa08_dataloader import FoldDataLoaders, create_fold_dataloaders
from etapa09_cnn1d import build_cnn1d_from_config
from etapa10_tcn import build_tcn_from_config
from etapa11_loss_metrics import (
    RegressionMetrics,
    TrainingConfig,
    build_criterion,
    build_optimizer,
    compute_batch_loss,
    evaluate_loader,
)


# ---------------------------------------------------------------------------
# 1. Histórico e early stopping
# ---------------------------------------------------------------------------
@dataclass
class FoldTrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_mae: list[float]
    best_epoch: int
    best_val_loss: float
    stop_reason: str = "max_epochs"


@dataclass
class FoldTrainingResult:
    fold_name: str
    val_subject_id: str
    model: nn.Module
    history: FoldTrainingHistory
    best_val_metrics: RegressionMetrics
    checkpoint_path: Path
    plot_path: Path
    metrics_path: Path
    y_true_scaled: np.ndarray | None = None
    y_pred_scaled: np.ndarray | None = None


class EarlyStopping:
    """
    Interrompe treino se val_loss não melhorar por ``patience`` épocas.

    Salva os pesos do melhor modelo em memória (state_dict).
    """

    def __init__(self, patience: int = 12, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_state: dict | None = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# 2. Loop de uma época de treino
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Uma passagem completa pelo DataLoader de treino.

    model.train()  -> dropout/batchnorm em modo treino
    zero_grad()    -> limpa gradientes acumulados
    backward()     -> calcula gradientes
    step()         -> atualiza pesos
    """
    model.train()
    losses: list[float] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = compute_batch_loss(criterion, y_hat, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(sum(losses) / len(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# 3. Preparar fold (normalização + janelas + loaders)
# ---------------------------------------------------------------------------
def build_model_from_config(config: ExperimentConfig, train_cfg: TrainingConfig | None = None) -> nn.Module:
    dropout = train_cfg.dropout if train_cfg else None
    if config.model_type == "cnn1d":
        return build_cnn1d_from_config(config, dropout=dropout)
    return build_tcn_from_config(config, dropout=dropout)


def prepare_loso_fold(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    val_subject_id: str = "02",
    dev_dataset: LoadedDataset | None = None,
) -> tuple[FoldDataLoaders, FoldNormalizationResult]:
    """Monta loaders de treino/val para um fold LOSO no grupo dev."""
    if dev_dataset is None:
        paths = create_output_dirs(config.output_dir)
        dataset = load_all_subjects(config.data_dir)
        split = load_subject_split(paths["splits"])
        dev_dataset, _ = apply_split_to_dataset(dataset, split)
    elif val_subject_id not in dev_dataset.subjects:
        raise KeyError(f"Sujeito {val_subject_id!r} não está no grupo de desenvolvimento.")

    fold = normalize_loso_fold(dev_dataset, val_subject_id=val_subject_id)

    train_batch = create_windows_from_dataset(
        fold.train_normalized,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=config.target_mode,
    )
    val_batch = create_windows_from_dataset(
        fold.val_normalized,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=config.target_mode,
    )

    loaders = create_fold_dataloaders(
        train_batch,
        val_batch,
        batch_size=train_cfg.batch_size,
        num_workers=getattr(train_cfg, "num_workers", 0),
    )
    return loaders, fold


# ---------------------------------------------------------------------------
# 4. Treinar um único fold
# ---------------------------------------------------------------------------
def train_single_fold(
    loaders: FoldDataLoaders,
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    fold: FoldNormalizationResult,
    device: torch.device | None = None,
    checkpoint_dir: Path | None = None,
    plots_dir: Path | None = None,
    metrics_dir: Path | None = None,
    artifact_prefix: str = "etapa12",
    verbose: bool = True,
    save_plot: bool = True,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    global_best_val_mae: float | None = None,
) -> FoldTrainingResult:
    """
    Treina um fold completo com early stopping.

    Critério de parada: val_loss (Huber/MSE/MAE conforme config).
    Melhor modelo: menor val_loss.
    """
    if device is None:
        device = get_device()

    paths = create_output_dirs(config.output_dir)
    checkpoint_dir = checkpoint_dir or paths["checkpoints"]
    plots_dir = plots_dir or paths["plots"]
    metrics_dir = metrics_dir or paths["metrics"]

    set_seed(config.seed)

    model = build_model_from_config(config, train_cfg).to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, train_cfg.patience // 3), min_lr=1e-6
    )
    early = EarlyStopping(patience=train_cfg.patience, min_delta=train_cfg.min_delta)

    history_train: list[float] = []
    history_val_loss: list[float] = []
    history_val_mae: list[float] = []
    stop_reason = "max_epochs"
    fold_best_val_mae = float("inf")

    for epoch in range(1, train_cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, loaders.train, criterion, optimizer, device)
        val_loss, val_metrics, _, _ = evaluate_loader(
            model, loaders.val, criterion, device
        )

        history_train.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_mae.append(val_metrics.mae)
        fold_best_val_mae = min(fold_best_val_mae, val_metrics.mae)

        scheduler.step(val_loss)
        early.step(val_loss, model, epoch)

        if verbose and (epoch == 1 or epoch % 10 == 0 or early.should_stop):
            print(
                f"  época {epoch:3d}/{train_cfg.max_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mae={val_metrics.mae:.4f}"
            )

        if enable_pruning and epoch >= pruning_warmup_epochs and global_best_val_mae is not None:
            if val_metrics.mae > global_best_val_mae * pruning_margin:
                stop_reason = "pruning"
                if verbose:
                    print(
                        f"  pruning na época {epoch}: val_mae={val_metrics.mae:.4f} > "
                        f"{global_best_val_mae * pruning_margin:.4f}"
                    )
                break

        if early.should_stop:
            stop_reason = "early_stopping"
            if verbose:
                print(f"  early stopping na época {epoch} (sem melhora por {train_cfg.patience} épocas)")
            break

    early.restore_best(model)
    best_val_loss, best_metrics, y_true, y_pred = evaluate_loader(
        model, loaders.val, criterion, device
    )

    fold_name = fold.fold_name
    ckpt_path = checkpoint_dir / f"{artifact_prefix}_{fold_name}_best.pt"
    plot_path = plots_dir / f"{artifact_prefix}_loss_{fold_name}.png"
    metrics_path = metrics_dir / f"{artifact_prefix}_{fold_name}_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "fold_name": fold_name,
            "val_subject_id": fold.val_subject_ids[0],
            "train_subject_ids": fold.train_subject_ids,
            "best_epoch": early.best_epoch,
            "best_val_loss": early.best_loss,
            "best_val_metrics": best_metrics.to_dict(),
        },
        ckpt_path,
    )

    fold.bundle.save(paths["scalers"] / f"{artifact_prefix}_{fold_name}_scaler.joblib")

    history = FoldTrainingHistory(
        train_loss=history_train,
        val_loss=history_val_loss,
        val_mae=history_val_mae,
        best_epoch=early.best_epoch,
        best_val_loss=early.best_loss,
        stop_reason=stop_reason,
    )

    if save_plot:
        plot_training_curves(history, plot_path, fold_name)

    metrics_payload = {
        "fold_name": fold_name,
        "val_subject_id": fold.val_subject_ids[0],
        "train_subject_ids": fold.train_subject_ids,
        "best_epoch": early.best_epoch,
        "best_val_loss": early.best_loss,
        "best_val_metrics": best_metrics.to_dict(),
        "n_epochs_ran": len(history_train),
        "stop_reason": stop_reason,
        "model_type": config.model_type,
        "loss": train_cfg.loss,
        "learning_rate": train_cfg.learning_rate,
        "dropout": train_cfg.dropout,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return FoldTrainingResult(
        fold_name=fold_name,
        val_subject_id=fold.val_subject_ids[0],
        model=model,
        history=history,
        best_val_metrics=best_metrics,
        checkpoint_path=ckpt_path,
        plot_path=plot_path,
        metrics_path=metrics_path,
        y_true_scaled=y_true,
        y_pred_scaled=y_pred,
    )


# ---------------------------------------------------------------------------
# 5. Gráfico de curvas de loss
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: FoldTrainingHistory,
    output_path: Path,
    fold_name: str,
) -> Path:
    epochs = range(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.train_loss, label="treino", color="steelblue")
    axes[0].plot(epochs, history.val_loss, label="validação", color="darkorange")
    axes[0].axvline(history.best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"melhor (ép. {history.best_epoch})")
    axes[0].set_xlabel("época")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"Loss — {fold_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.val_mae, label="val MAE", color="seagreen")
    axes[1].axvline(history.best_epoch, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("época")
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"MAE validação — {fold_name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 6. Relatório legível
# ---------------------------------------------------------------------------
def print_training_report(result: FoldTrainingResult, train_cfg: TrainingConfig) -> None:
    h = result.history
    m = result.best_val_metrics

    print("=" * 60)
    print("ETAPA 12 — Treinamento de um único fold")
    print("=" * 60)
    print(f"Fold               : {result.fold_name}")
    print(f"Validação (LOSO)   : sujeito {result.val_subject_id}")
    print(f"Épocas executadas  : {len(h.train_loss)}")
    print(f"Melhor época       : {h.best_epoch}")
    print(f"Melhor val_loss    : {h.best_val_loss:.4f}")
    print(f"Val MAE (melhor)   : {m.mae:.4f}  (escala vicon normalizada)")
    print(f"Val RMSE           : {m.rmse:.4f}")
    print(f"Val R²             : {m.r2:.4f}")
    print(f"Val Bias           : {m.bias:+.4f}")

    print("\nArquivos salvos:")
    print(f"  checkpoint : {result.checkpoint_path}")
    print(f"  plot loss  : {result.plot_path}")
    print(f"  métricas   : {result.metrics_path}")

    print("\nInterpretação rápida das curvas:")
    print("  - treino ↓ e val ↓ juntos    : aprendizado saudável")
    print("  - treino ↓, val ↑ ou estagnado : possível overfitting")
    print("  - early stopping evita treinar além do melhor val")

    print("\nPróxima etapa: LOSO completo nos 70% (Etapa 13).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 7. Ponto de entrada da Etapa 12
# ---------------------------------------------------------------------------
def run_stage12_train_fold(
    config: ExperimentConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    val_subject_id: str = "02",
) -> FoldTrainingResult:
    """
    Treina um único fold LOSO no grupo de desenvolvimento.

    Usa apenas sujeitos dev. Não toca no teste final (30%).
    """
    if config is None:
        config = build_default_config()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    device = get_device()
    print(f"Dispositivo: {device}")
    print(f"Modelo: {config.model_type.upper()} | Loss: {train_cfg.loss.upper()}")
    print(f"Preparando fold LOSO (val={val_subject_id})...")

    loaders, fold = prepare_loso_fold(config, train_cfg, val_subject_id=val_subject_id)

    print(f"Treino: {len(loaders.train_dataset)} janelas | Val: {len(loaders.val_dataset)} janelas")
    print("Iniciando treino...")

    result = train_single_fold(
        loaders=loaders,
        config=config,
        train_cfg=train_cfg,
        fold=fold,
        device=device,
    )

    print_training_report(result, train_cfg)
    return result


if __name__ == "__main__":
    run_stage12_train_fold()
