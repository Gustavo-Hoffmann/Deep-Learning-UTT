#!/usr/bin/env python3
"""
UTT / smartphone -> Vicon displacement regression (robust version with LOSO)

What this script does
---------------------
1) Loads one or many CSV/XLSX files containing time, 6 smartphone channels and 1 Vicon target.
2) Builds sliding windows for temporal learning.
3) Trains a deep-learning model (CNN + BiLSTM + attention pooling) for regression.
4) Evaluates using either:
   - LOSO (leave-one-subject-out), or
   - subject-level 80/20, or
   - chronological 80/20, or
   - random 80/20 over all windows from all subjects combined (default).
5) Saves metrics, predictions, plots, and model weights.

Designed for cases where instantaneous acceleration is not enough and temporal context matters.

Example expected columns (matching the uploaded example file):
- tempo_norm_s
- accX_m_s2
- accY_m_s2
- accZ_m_s2
- gyroX_rad_s
- gyroY_rad_s
- gyroZ_rad_s
- vicon_esternoZ_mm_norm

Typical usage
-------------
python utt_deep_learning_vicon_loso.py \
  --input "/path/to/data/*.xlsx" \
  --out "/path/to/output" \
  --split loso \
  --time-col tempo_norm_s \
  --target-col vicon_esternoZ_mm_norm \
  --feature-cols accX_m_s2 accY_m_s2 accZ_m_s2 gyroX_rad_s gyroY_rad_s gyroZ_rad_s

Notes
-----
- If there is no subject column, the script uses the filename stem as subject_id.
- If there is only one subject and split=loso, it falls back to chronological_80_20 unless --strict-loso is used.
- Feature scaling and target scaling are fit on train only, fold by fold.
- Windowing is performed AFTER split, to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

_mpl_cache = Path(__file__).resolve().parent / ".mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Training monitor helpers
# -----------------------------
def _fmt_hms(seconds: float) -> str:
    """Format seconds as H:MM:SS (non-negative)."""
    try:
        seconds = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return "--:--:--"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _is_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


def _render_progress(prefix: str, i: int, total: int, running_loss: float, t_start: float) -> None:
    """Inline progress bar rendering (only called when stdout is a TTY)."""
    bar_len = 24
    done = i / total if total else 1.0
    done = min(max(done, 0.0), 1.0)
    filled = int(bar_len * done)
    bar = "#" * filled + "-" * (bar_len - filled)
    elapsed = time.perf_counter() - t_start
    eta = (elapsed / done * (1.0 - done)) if done > 0 else 0.0
    it_s = i / elapsed if elapsed > 0 else 0.0
    sys.stdout.write(
        f"\r{prefix} [{bar}] {i:>4d}/{total:<4d} "
        f"loss={running_loss:.4f} it/s={it_s:5.1f} "
        f"elapsed={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)}   "
    )
    sys.stdout.flush()


# -----------------------------
# Helpers / config
# -----------------------------
@dataclass
class TrainConfig:
    input: str = "Output_ML/*.xlsx"
    out: str = "runs/all_subjects_80_20"
    split: str = "random_80_20"  # loso | subject_80_20 | chronological_80_20 | random_80_20
    time_col: str = "tempo_norm_s"
    target_col: str = "vicon_esternoZ_mm_norm"
    subject_col: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    file_id_as_subject: bool = True
    strict_loso: bool = False

    window: int = 60
    stride: int = 1
    horizon: int = 0  # predict target at end-of-window + horizon

    batch_size: int = 256
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.20
    hidden_size: int = 64
    lstm_layers: int = 1

    train_fraction: float = 0.80
    val_fraction: float = 0.10
    random_seed: int = 42

    add_engineered_features: bool = True
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_FEATURE_COLS = [
    "accX_m_s2",
    "accY_m_s2",
    "accZ_m_s2",
    "gyroX_rad_s",
    "gyroY_rad_s",
    "gyroZ_rad_s",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading + engineering
# -----------------------------
def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def infer_subject_id(df: pd.DataFrame, path: Path, subject_col: Optional[str]) -> pd.Series:
    if subject_col and subject_col in df.columns:
        return df[subject_col].astype(str)
    return pd.Series([path.stem] * len(df), index=df.index, dtype="object")


def compute_dt(time_values: np.ndarray) -> np.ndarray:
    dt = np.diff(time_values, prepend=time_values[0])
    if len(dt) > 1:
        positive_dt = dt[dt > 0]
        fallback = np.median(positive_dt) if len(positive_dt) else 1.0 / 60.0
    else:
        fallback = 1.0 / 60.0
    dt = np.where(dt <= 0, fallback, dt)
    return dt


def add_engineered_features(df: pd.DataFrame, time_col: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    acc_cols = [c for c in feature_cols if c.lower().startswith("acc")]
    gyro_cols = [c for c in feature_cols if c.lower().startswith("gyro")]

    if len(acc_cols) >= 3:
        acc = df[acc_cols].to_numpy(dtype=float)
        df["acc_norm"] = np.linalg.norm(acc, axis=1)
        # crude integrated features can help the network without forcing pure integration end-to-end
        if "accY_m_s2" in df.columns and time_col in df.columns:
            t = df[time_col].to_numpy(dtype=float)
            dt = compute_dt(t)
            ay = df["accY_m_s2"].to_numpy(dtype=float)
            ay_centered = ay - np.nanmedian(ay)
            vel_proxy = np.cumsum(ay_centered * dt)
            disp_proxy = np.cumsum(vel_proxy * dt)
            df["accY_vel_proxy"] = vel_proxy
            df["accY_disp_proxy"] = disp_proxy

    if len(gyro_cols) >= 3:
        gyro = df[gyro_cols].to_numpy(dtype=float)
        df["gyro_norm"] = np.linalg.norm(gyro, axis=1)

    return df


def load_all_data(cfg: TrainConfig) -> pd.DataFrame:
    input_pattern = cfg.input
    has_glob = any(ch in input_pattern for ch in "*?[")
    if has_glob:
        pattern_path = Path(input_pattern)
        if pattern_path.is_absolute():
            paths = sorted(Path(pattern_path.anchor).glob(str(pattern_path.relative_to(pattern_path.anchor))))
        else:
            paths = sorted(Path().glob(input_pattern))
            if not paths:
                # Fallback: resolve relative to this script's directory
                script_dir = Path(__file__).resolve().parent
                paths = sorted(script_dir.glob(input_pattern))
    else:
        paths = [Path(input_pattern)]
    if not paths:
        raise FileNotFoundError(f"No files matched: {cfg.input}")

    frames: List[pd.DataFrame] = []
    for path in paths:
        df = read_table(path)
        df = df.copy()
        df["source_file"] = path.name
        df["subject_id"] = infer_subject_id(df, path, cfg.subject_col)

        required = [cfg.time_col, cfg.target_col] + list(cfg.feature_cols or DEFAULT_FEATURE_COLS)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path.name}: {missing}")

        if cfg.add_engineered_features:
            df = add_engineered_features(df, cfg.time_col, cfg.feature_cols or DEFAULT_FEATURE_COLS)

        frames.append(df)

    data = pd.concat(frames, axis=0, ignore_index=True)
    data = data.sort_values(["subject_id", cfg.time_col]).reset_index(drop=True)
    return data


# -----------------------------
# Windowing
# -----------------------------
def make_windows_from_subject(
    df_sub: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    window: int,
    stride: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X [N, T, F], y [N], times [N]."""
    values_x = df_sub[list(feature_cols)].to_numpy(dtype=np.float32)
    values_y = df_sub[target_col].to_numpy(dtype=np.float32)
    times = df_sub.iloc[:, 0].to_numpy()  # caller later replaces if needed only for export/diagnostics

    last_start = len(df_sub) - window - horizon
    if last_start < 0:
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    X_list, y_list, t_list = [], [], []
    for start in range(0, last_start + 1, stride):
        end = start + window
        label_idx = end - 1 + horizon
        X_list.append(values_x[start:end])
        y_list.append(values_y[label_idx])
        t_list.append(times[label_idx])

    return np.stack(X_list), np.asarray(y_list), np.asarray(t_list)


def make_windows(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    time_col: str,
    target_col: str,
    window: int,
    stride: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    Xs, ys, times, subjects = [], [], [], []
    for subject_id, df_sub in data.groupby("subject_id", sort=True):
        df_sub = df_sub.sort_values(time_col).reset_index(drop=True)
        X, y, t = make_windows_from_subject(df_sub, feature_cols, target_col, window, stride, horizon)
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        times.append(t)
        subjects.append(np.asarray([subject_id] * len(y), dtype=object))

    if not Xs:
        raise ValueError("No windows could be created. Reduce window/horizon or check file lengths.")

    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "time": np.concatenate(times, axis=0),
        "subject_id": np.concatenate(subjects, axis=0),
    }


# -----------------------------
# Scaling
# -----------------------------
def fit_transform_windows(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    n_feat = X_train.shape[-1]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_feat)
    X_val_2d = X_val.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)

    X_train_scaled = x_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
    )


# -----------------------------
# Torch dataset/model
# -----------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class AttentionPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)  # [B, T]
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        conv_channels = 64
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = AttentionPool(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)           # [B, F, T]
        x = self.conv(x)                # [B, C, T]
        x = x.transpose(1, 2)           # [B, T, C]
        x, _ = self.lstm(x)             # [B, T, 2H]
        x = self.attn(x)                # [B, 2H]
        x = self.head(x).squeeze(-1)    # [B]
        return x


# -----------------------------
# Training / evaluation
# -----------------------------
def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val)
    test_ds = WindowDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str,
    train: bool = True,
    progress_prefix: Optional[str] = None,
) -> float:
    model.train(train)
    losses: List[float] = []
    n_batches = len(loader)
    show_bar = progress_prefix is not None and _is_tty() and n_batches > 0
    t0 = time.perf_counter()

    update_every = max(1, n_batches // 20) if n_batches else 1

    for i, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.set_grad_enabled(train):
            preds = model(xb)
            loss = criterion(preds, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        losses.append(loss.item())

        if show_bar and (i == n_batches or i % update_every == 0):
            _render_progress(
                prefix=progress_prefix or "",
                i=i,
                total=n_batches,
                running_loss=float(np.mean(losses)),
                t_start=t0,
            )

    if show_bar:
        sys.stdout.write("\r" + " " * 120 + "\r")  # erase bar line
        sys.stdout.flush()

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model, loader, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        yhat = model(xb).detach().cpu().numpy()
        preds.append(yhat)
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


def inverse_transform_target(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": corr}


def train_one_fold(
    fold_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_subject_ids: np.ndarray,
    test_times: np.ndarray,
    cfg: TrainConfig,
    out_dir: Path,
) -> Dict[str, float]:
    (
        X_train_s, y_train_s,
        X_val_s, y_val_s,
        X_test_s, y_test_s,
        x_scaler, y_scaler,
    ) = fit_transform_windows(X_train, y_train, X_val, y_val, X_test, y_test)

    train_loader, val_loader, test_loader = make_loaders(
        X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    )

    model = CNNBiLSTMRegressor(
        n_features=X_train.shape[-1],
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    criterion = nn.HuberLoss(delta=1.0)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    stale = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(
        f"[{fold_name}] training start | device={cfg.device} | "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)} | "
        f"max_epochs={cfg.epochs} patience={cfg.patience} | trainable_params={n_params:,}"
    )

    train_start = time.perf_counter()
    epoch_durations: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        ep_start = time.perf_counter()

        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, cfg.device, train=True,
            progress_prefix=f"  [{fold_name}] ep {epoch:>3d}/{cfg.epochs} train",
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, cfg.device, train=False,
            progress_prefix=f"  [{fold_name}] ep {epoch:>3d}/{cfg.epochs}  val ",
        )
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        ep_dur = time.perf_counter() - ep_start
        epoch_durations.append(ep_dur)
        avg_ep = float(np.mean(epoch_durations[-5:]))  # moving average (last 5)
        total_elapsed = time.perf_counter() - train_start
        remaining_epochs = cfg.epochs - epoch
        eta_full = avg_ep * remaining_epochs
        eta_patience = avg_ep * max(0, cfg.patience - stale)
        eta = min(eta_full, eta_patience) if not improved else eta_full
        current_lr = optimizer.param_groups[0]["lr"]
        flag = "*" if improved else " "

        _log(
            f"[{fold_name}] ep {epoch:>3d}/{cfg.epochs} {flag} "
            f"train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.2e} | "
            f"ep={_fmt_hms(ep_dur)} total={_fmt_hms(total_elapsed)} ETA~{_fmt_hms(eta)} | "
            f"stale={stale}/{cfg.patience} best={best_val:.4f}@ep{best_epoch}"
        )

        if stale >= cfg.patience:
            _log(f"[{fold_name}] early stopping triggered at epoch {epoch} (stale={stale})")
            break

    if best_state is None:
        raise RuntimeError(f"Training failed on fold {fold_name}: no best state recorded")

    train_total = time.perf_counter() - train_start
    _log(
        f"[{fold_name}] training done in {_fmt_hms(train_total)} | "
        f"best_val={best_val:.4f} @ ep {best_epoch} | epochs_run={len(epoch_durations)}"
    )

    model.load_state_dict(best_state)

    pred_test_scaled = predict(model, test_loader, cfg.device)
    pred_test = inverse_transform_target(pred_test_scaled, y_scaler)
    y_test_real = y_test

    metrics = regression_metrics(y_test_real, pred_test)
    metrics.update({"fold": fold_name, "best_epoch": best_epoch, "n_test": int(len(y_test_real))})

    # Save fold artifacts
    fold_dir = out_dir / fold_name
    ensure_dir(fold_dir)

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    torch.save(model.state_dict(), fold_dir / "best_model.pt")

    with open(fold_dir / "x_scaler_mean_std.json", "w", encoding="utf-8") as f:
        json.dump({"mean": x_scaler.mean_.tolist(), "scale": x_scaler.scale_.tolist()}, f, indent=2)
    with open(fold_dir / "y_scaler_mean_std.json", "w", encoding="utf-8") as f:
        json.dump({"mean": y_scaler.mean_.tolist(), "scale": y_scaler.scale_.tolist()}, f, indent=2)

    preds_df = pd.DataFrame({
        "fold": fold_name,
        "subject_id": test_subject_ids,
        "time": test_times,
        "y_true": y_test_real,
        "y_pred": pred_test,
        "abs_error": np.abs(y_test_real - pred_test),
    })
    preds_df.to_csv(fold_dir / "predictions.csv", index=False)

    plot_training_history(history, fold_dir / "history.png")
    plot_true_vs_pred(y_test_real, pred_test, fold_dir / "true_vs_pred.png", title=f"{fold_name} | True vs Pred")
    plot_scatter(y_test_real, pred_test, fold_dir / "scatter.png", title=f"{fold_name} | Scatter")

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# -----------------------------
# Plotting
# -----------------------------
def plot_training_history(history: List[Dict[str, float]], out_path: Path) -> None:
    df = pd.DataFrame(history)
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "") -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="y_true")
    plt.plot(y_pred, label="y_pred")
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = "") -> None:
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Splits
# -----------------------------
def split_subjects_80_20(subject_ids: Sequence[str], train_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    unique_subjects = sorted(set(subject_ids))
    rng = random.Random(seed)
    shuffled = unique_subjects[:]
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_fraction)))
    n_train = min(n_train, max(1, len(shuffled) - 1))
    train_subjects = shuffled[:n_train]
    test_subjects = shuffled[n_train:]
    if not test_subjects:
        test_subjects = [train_subjects.pop()]
    return train_subjects, test_subjects


def subject_level_train_val_split(train_subjects: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    subjects = list(sorted(set(train_subjects)))
    if len(subjects) <= 2:
        return subjects, []
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = subjects[:n_val]
    train_subjects_final = subjects[n_val:]
    return train_subjects_final, val_subjects


def chronological_split_windows(windows: Dict[str, np.ndarray], train_fraction: float, val_fraction: float) -> Dict[str, Dict[str, np.ndarray]]:
    X = windows["X"]
    y = windows["y"]
    t = windows["time"]
    s = windows["subject_id"]

    order = np.lexsort((t, s))
    X, y, t, s = X[order], y[order], t[order], s[order]

    n = len(y)
    n_train = max(1, int(n * train_fraction))
    remaining = n - n_train
    n_val = max(1, int(remaining * (val_fraction / max(1e-8, 1.0 - train_fraction)))) if remaining >= 3 else max(1, remaining // 2)
    n_val = min(n_val, max(1, n - n_train - 1)) if n - n_train >= 2 else 0

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    return {
        "train": {"X": X[idx_train], "y": y[idx_train], "time": t[idx_train], "subject_id": s[idx_train]},
        "val": {"X": X[idx_val], "y": y[idx_val], "time": t[idx_val], "subject_id": s[idx_val]},
        "test": {"X": X[idx_test], "y": y[idx_test], "time": t[idx_test], "subject_id": s[idx_test]},
    }


def random_split_windows(
    windows: Dict[str, np.ndarray],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Shuffle windows from all subjects combined and split into train/val/test.

    - `train_fraction` of all windows go to train+val (with `val_fraction` of the
      full dataset going to validation; the rest of the 80% goes to training).
    - The remaining (1 - train_fraction) goes to test.
    """
    X = windows["X"]
    y = windows["y"]
    t = windows["time"]
    s = windows["subject_id"]

    n = len(y)
    if n < 3:
        raise ValueError("Not enough windows for a random 80/20 split.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X, y, t, s = X[perm], y[perm], t[perm], s[perm]

    n_test = max(1, int(round(n * (1.0 - train_fraction))))
    n_test = min(n_test, n - 2)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - n_test - 1))
    n_train = n - n_test - n_val

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)

    return {
        "train": {"X": X[idx_train], "y": y[idx_train], "time": t[idx_train], "subject_id": s[idx_train]},
        "val":   {"X": X[idx_val],   "y": y[idx_val],   "time": t[idx_val],   "subject_id": s[idx_val]},
        "test":  {"X": X[idx_test],  "y": y[idx_test],  "time": t[idx_test],  "subject_id": s[idx_test]},
    }


def subset_windows_by_subject(windows: Dict[str, np.ndarray], keep_subjects: Sequence[str]) -> Dict[str, np.ndarray]:
    keep_subjects = set(map(str, keep_subjects))
    mask = np.array([str(s) in keep_subjects for s in windows["subject_id"]])
    return {k: v[mask] for k, v in windows.items()}


def maybe_split_train_val_within_train_subjects(
    train_windows: Dict[str, np.ndarray],
    val_fraction: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train_subjects = sorted(set(map(str, train_windows["subject_id"])))
    if len(train_subjects) >= 4:
        train_subs, val_subs = subject_level_train_val_split(train_subjects, val_fraction, seed)
        if val_subs:
            return subset_windows_by_subject(train_windows, train_subs), subset_windows_by_subject(train_windows, val_subs)

    # Fallback: chronological split inside the train windows
    X = train_windows["X"]
    y = train_windows["y"]
    t = train_windows["time"]
    s = train_windows["subject_id"]
    order = np.lexsort((t, s))
    X, y, t, s = X[order], y[order], t[order], s[order]
    n = len(y)
    if n < 10:
        # tiny dataset fallback
        n_val = max(1, n // 5)
    else:
        n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - 1))
    n_train = n - n_val
    return (
        {"X": X[:n_train], "y": y[:n_train], "time": t[:n_train], "subject_id": s[:n_train]},
        {"X": X[n_train:], "y": y[n_train:], "time": t[n_train:], "subject_id": s[n_train:]},
    )


# -----------------------------
# Main evaluation routines
# -----------------------------
def run_loso(data: pd.DataFrame, feature_cols: Sequence[str], cfg: TrainConfig, out_dir: Path) -> pd.DataFrame:
    subjects = sorted(set(map(str, data["subject_id"])))
    if len(subjects) < 2:
        if cfg.strict_loso:
            raise ValueError("LOSO requires at least 2 subjects.")
        print("[WARN] Only one subject found. Falling back to chronological_80_20.")
        return run_chronological_80_20(data, feature_cols, cfg, out_dir)

    all_metrics = []
    for fold_idx, test_subject in enumerate(subjects, start=1):
        fold_name = f"fold_{fold_idx:02d}_test_{test_subject}"
        train_df = data[data["subject_id"].astype(str) != str(test_subject)].copy()
        test_df = data[data["subject_id"].astype(str) == str(test_subject)].copy()

        train_windows = make_windows(train_df, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
        test_windows = make_windows(test_df, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
        train_windows, val_windows = maybe_split_train_val_within_train_subjects(train_windows, cfg.val_fraction, cfg.random_seed + fold_idx)

        metrics = train_one_fold(
            fold_name=fold_name,
            X_train=train_windows["X"], y_train=train_windows["y"],
            X_val=val_windows["X"], y_val=val_windows["y"],
            X_test=test_windows["X"], y_test=test_windows["y"],
            test_subject_ids=test_windows["subject_id"], test_times=test_windows["time"],
            cfg=cfg,
            out_dir=out_dir,
        )
        all_metrics.append(metrics)
        print(f"[LOSO] {fold_name}: {metrics}")

    return pd.DataFrame(all_metrics)


def run_subject_80_20(data: pd.DataFrame, feature_cols: Sequence[str], cfg: TrainConfig, out_dir: Path) -> pd.DataFrame:
    subjects = sorted(set(map(str, data["subject_id"])))
    if len(subjects) < 2:
        print("[WARN] subject_80_20 requested but only one subject found. Falling back to chronological_80_20.")
        return run_chronological_80_20(data, feature_cols, cfg, out_dir)

    train_subs, test_subs = split_subjects_80_20(subjects, cfg.train_fraction, cfg.random_seed)
    train_df = data[data["subject_id"].astype(str).isin(train_subs)].copy()
    test_df = data[data["subject_id"].astype(str).isin(test_subs)].copy()

    train_windows = make_windows(train_df, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
    test_windows = make_windows(test_df, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
    train_windows, val_windows = maybe_split_train_val_within_train_subjects(train_windows, cfg.val_fraction, cfg.random_seed)

    metrics = train_one_fold(
        fold_name="subject_80_20",
        X_train=train_windows["X"], y_train=train_windows["y"],
        X_val=val_windows["X"], y_val=val_windows["y"],
        X_test=test_windows["X"], y_test=test_windows["y"],
        test_subject_ids=test_windows["subject_id"], test_times=test_windows["time"],
        cfg=cfg,
        out_dir=out_dir,
    )
    return pd.DataFrame([metrics])


def run_random_80_20(data: pd.DataFrame, feature_cols: Sequence[str], cfg: TrainConfig, out_dir: Path) -> pd.DataFrame:
    """Train a single model using an 80/20 random split over windows from ALL subjects pooled together."""
    t_win = time.perf_counter()
    windows = make_windows(data, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
    _log(
        f"windows built in {_fmt_hms(time.perf_counter() - t_win)} | "
        f"total_windows={len(windows['y']):,} window={cfg.window} stride={cfg.stride} horizon={cfg.horizon} "
        f"n_features={windows['X'].shape[-1]}"
    )
    split = random_split_windows(windows, cfg.train_fraction, cfg.val_fraction, cfg.random_seed)

    n_subjects_train = len(set(map(str, split["train"]["subject_id"])))
    n_subjects_test = len(set(map(str, split["test"]["subject_id"])))
    _log(
        f"[random_80_20] train={len(split['train']['y']):,} val={len(split['val']['y']):,} "
        f"test={len(split['test']['y']):,} | "
        f"subjects_in_train={n_subjects_train} subjects_in_test={n_subjects_test}"
    )

    metrics = train_one_fold(
        fold_name="random_80_20",
        X_train=split["train"]["X"], y_train=split["train"]["y"],
        X_val=split["val"]["X"], y_val=split["val"]["y"],
        X_test=split["test"]["X"], y_test=split["test"]["y"],
        test_subject_ids=split["test"]["subject_id"], test_times=split["test"]["time"],
        cfg=cfg,
        out_dir=out_dir,
    )
    return pd.DataFrame([metrics])


def run_chronological_80_20(data: pd.DataFrame, feature_cols: Sequence[str], cfg: TrainConfig, out_dir: Path) -> pd.DataFrame:
    windows = make_windows(data, feature_cols, cfg.time_col, cfg.target_col, cfg.window, cfg.stride, cfg.horizon)
    split = chronological_split_windows(windows, cfg.train_fraction, cfg.val_fraction)

    metrics = train_one_fold(
        fold_name="chronological_80_20",
        X_train=split["train"]["X"], y_train=split["train"]["y"],
        X_val=split["val"]["X"], y_val=split["val"]["y"],
        X_test=split["test"]["X"], y_test=split["test"]["y"],
        test_subject_ids=split["test"]["subject_id"], test_times=split["test"]["time"],
        cfg=cfg,
        out_dir=out_dir,
    )
    return pd.DataFrame([metrics])


# -----------------------------
# Reports
# -----------------------------
def summarize_and_save(metrics_df: pd.DataFrame, out_dir: Path, cfg: TrainConfig, data: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    metrics_df.to_csv(out_dir / "metrics_all_folds.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "n_rows": int(len(data)),
        "n_subjects": int(data["subject_id"].nunique()),
        "subjects": sorted(map(str, data["subject_id"].unique().tolist())),
        "feature_cols": list(feature_cols),
        "target_col": cfg.target_col,
        "time_col": cfg.time_col,
        "metrics_mean": metrics_df[["rmse", "mae", "r2", "pearson_r"]].mean(numeric_only=True).to_dict(),
        "metrics_std": metrics_df[["rmse", "mae", "r2", "pearson_r"]].std(numeric_only=True).fillna(0).to_dict(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # aggregate chart if multiple folds
    if len(metrics_df) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(metrics_df["fold"], metrics_df["rmse"], marker="o", label="RMSE")
        plt.plot(metrics_df["fold"], metrics_df["mae"], marker="o", label="MAE")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Error")
        plt.title("Fold-wise errors")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "fold_errors.png", dpi=150)
        plt.close()

    # concatenate all predictions if available
    pred_files = list(out_dir.glob("*/predictions.csv"))
    if pred_files:
        all_preds = pd.concat([pd.read_csv(p) for p in pred_files], axis=0, ignore_index=True)
        all_preds.to_csv(out_dir / "predictions_all_folds.csv", index=False)
        plot_true_vs_pred(all_preds["y_true"].to_numpy(), all_preds["y_pred"].to_numpy(), out_dir / "all_true_vs_pred.png", title="All folds | True vs Pred")
        plot_scatter(all_preds["y_true"].to_numpy(), all_preds["y_pred"].to_numpy(), out_dir / "all_scatter.png", title="All folds | Scatter")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Robust deep learning pipeline for smartphone -> Vicon regression with LOSO support.")
    parser.add_argument("--input", type=str, default="Output_ML/*.xlsx",
                        help="Input file path or glob pattern. Default: 'Output_ML/*.xlsx' (all subjects).")
    parser.add_argument("--out", type=str, default="runs/all_subjects_80_20",
                        help="Output directory. Default: 'runs/all_subjects_80_20'.")
    parser.add_argument("--split", type=str, default="random_80_20",
                        choices=["loso", "subject_80_20", "chronological_80_20", "random_80_20"],
                        help="Evaluation split. Default: random_80_20 (pool all subjects, random 80/20).")
    parser.add_argument("--time-col", type=str, default="tempo_norm_s")
    parser.add_argument("--target-col", type=str, default="vicon_esternoZ_mm_norm")
    parser.add_argument("--subject-col", type=str, default=None)
    parser.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLS)
    parser.add_argument("--strict-loso", action="store_true")

    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)

    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-engineered-features", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    cfg = TrainConfig(
        input=args.input,
        out=args.out,
        split=args.split,
        time_col=args.time_col,
        target_col=args.target_col,
        subject_col=args.subject_col,
        feature_cols=args.feature_cols,
        strict_loso=args.strict_loso,
        window=args.window,
        stride=args.stride,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
        add_engineered_features=not args.no_engineered_features,
        num_workers=args.num_workers,
        device=args.device,
    )
    return cfg


def main() -> None:
    run_start = time.perf_counter()
    cfg = parse_args()
    set_seed(cfg.random_seed)

    out_dir = Path(cfg.out)
    ensure_dir(out_dir)
    feature_cols = list(cfg.feature_cols or DEFAULT_FEATURE_COLS)

    _log(f"starting run | split={cfg.split} | device={cfg.device} | out={out_dir.resolve()}")
    _log(f"loading data from: {cfg.input}")
    t_load = time.perf_counter()
    data = load_all_data(cfg)
    _log(
        f"data loaded in {_fmt_hms(time.perf_counter() - t_load)} | "
        f"rows={len(data):,} subjects={data['subject_id'].nunique()} files={data['source_file'].nunique()}"
    )

    # Keep only columns needed + engineered features
    final_feature_cols = feature_cols[:]
    if cfg.add_engineered_features:
        for extra in ["acc_norm", "gyro_norm", "accY_vel_proxy", "accY_disp_proxy"]:
            if extra in data.columns:
                final_feature_cols.append(extra)

    # Basic cleaning
    keep_cols = [cfg.time_col, cfg.target_col, "subject_id", "source_file"] + final_feature_cols
    data = data[keep_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    data.head(20).to_csv(out_dir / "data_preview.csv", index=False)

    if cfg.split == "loso":
        metrics_df = run_loso(data, final_feature_cols, cfg, out_dir)
    elif cfg.split == "subject_80_20":
        metrics_df = run_subject_80_20(data, final_feature_cols, cfg, out_dir)
    elif cfg.split == "chronological_80_20":
        metrics_df = run_chronological_80_20(data, final_feature_cols, cfg, out_dir)
    elif cfg.split == "random_80_20":
        metrics_df = run_random_80_20(data, final_feature_cols, cfg, out_dir)
    else:
        raise ValueError(f"Unsupported split mode: {cfg.split}")

    summarize_and_save(metrics_df, out_dir, cfg, data, final_feature_cols)

    total_runtime = time.perf_counter() - run_start
    _log(f"DONE | total runtime {_fmt_hms(total_runtime)} | output={out_dir.resolve()}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()