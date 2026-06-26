#!/usr/bin/env python3
"""Loop de treinamento com early stopping, scheduler e cronômetro."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ScalerBundle, WindowDataset
from .losses import CompositeCurveLoss
from .models import ResidualTCN
from .utils import EpochTimer, amp_supported, format_duration


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_epochs: int = 250
    patience: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    quick: bool = False


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    best_val_rmse_cm: float
    history: list[dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0
    best_model_path: Path | None = None


def _collate(batch: list[dict]) -> dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "vicon": torch.stack([b["vicon"] for b in batch]),
        "calibrado": torch.stack([b["calibrado"] for b in batch]),
        "residual": torch.stack([b["residual"] for b in batch]),
    }


def _batch_rmse_cm(pred_dl: torch.Tensor, vicon: torch.Tensor) -> float:
    err = (pred_dl - vicon).detach().cpu().numpy()
    return float(np.sqrt(np.mean(err**2)))


def train_model(
    model: ResidualTCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scalers: ScalerBundle,
    *,
    device: torch.device,
    backend: str,
    out_dir: Path,
    cfg: TrainConfig,
) -> TrainResult:
    if cfg.quick:
        cfg = TrainConfig(
            batch_size=min(cfg.batch_size, 16),
            max_epochs=20,
            patience=8,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            num_workers=cfg.num_workers,
            quick=True,
        )

    criterion = CompositeCurveLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    use_amp = amp_supported(device, backend)  # type: ignore[arg-type]
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.to(device)
    best_val = float("inf")
    best_rmse = float("inf")
    best_epoch = 0
    patience_ctr = 0
    history: list[dict[str, float]] = []
    epoch_timer = EpochTimer(total_epochs=cfg.max_epochs)
    import time

    wall_t0 = time.perf_counter()
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_model.pt"
    log_rows: list[dict] = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            x = batch["x"].to(device)
            vicon = batch["vicon"].to(device)
            calibrado = batch["calibrado"].to(device)
            resid_scaled = batch["residual"].to(device)

            optimizer.zero_grad(set_to_none=True)

            resid_pred_scaled = model(x)
            if scalers.residual_scaler is not None:
                scale = float(scalers.residual_scaler.scale_[0])
                mean = float(scalers.residual_scaler.mean_[0])
                resid_pred = resid_pred_scaled * scale + mean
            else:
                resid_pred = resid_pred_scaled
            pred_dl = calibrado + resid_pred

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, _ = criterion(pred_dl, vicon)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss, _ = criterion(pred_dl, vicon)
                loss.backward()
                optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        val_rmses: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                vicon = batch["vicon"].to(device)
                calibrado = batch["calibrado"].to(device)

                resid_pred_scaled = model(x)
                if scalers.residual_scaler is not None:
                    scale = float(scalers.residual_scaler.scale_[0])
                    mean = float(scalers.residual_scaler.mean_[0])
                    resid_pred = resid_pred_scaled * scale + mean
                else:
                    resid_pred = resid_pred_scaled
                pred_dl = calibrado + resid_pred
                loss, _ = criterion(pred_dl, vicon)
                val_losses.append(float(loss.detach().cpu()))
                val_rmses.append(_batch_rmse_cm(pred_dl, vicon))

        tr_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        va_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        va_rmse = float(np.mean(val_rmses)) if val_rmses else float("nan")
        scheduler.step(va_loss)

        epoch_dur = epoch_timer.mark_epoch_end()
        elapsed = epoch_timer.elapsed_total()
        eta = epoch_timer.eta(epoch) or 0.0

        print(
            f"Epoch {epoch:03d}/{cfg.max_epochs} | "
            f"train_loss={tr_loss:.5f} | val_loss={va_loss:.5f} | "
            f"val_RMSE={va_rmse:.3f} cm | epoch_time={epoch_dur:.1f}s | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)} | "
            f"device={device}"
        )

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_rmse_cm": va_rmse,
            "epoch_time_s": epoch_dur,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        log_rows.append(row)

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_rmse = va_rmse
            best_epoch = epoch
            patience_ctr = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": va_loss,
                    "val_rmse_cm": va_rmse,
                },
                best_path,
            )
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"Early stopping na época {epoch} (melhor={best_epoch}).")
                break

    train_seconds = time.perf_counter() - wall_t0

    pd.DataFrame(log_rows).to_csv(out_dir / "training_time_log.csv", index=False)
    with open(out_dir / "training_time_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"train_seconds={train_seconds:.2f}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val:.6f}\n")
        f.write(f"best_val_rmse_cm={best_rmse:.4f}\n")
        f.write(f"epochs_run={len(history)}\n")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        best_val_rmse_cm=best_rmse,
        history=history,
        train_seconds=train_seconds,
        best_model_path=best_path if best_path.exists() else None,
    )


def build_loaders(
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        drop_last=len(train_ds) >= cfg.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
    )
    return train_loader, val_loader
