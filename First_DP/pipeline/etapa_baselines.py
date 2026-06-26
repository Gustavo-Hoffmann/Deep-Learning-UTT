#!/usr/bin/env python3
"""Baselines obrigatórios no LOSO dos 70% (--baselines)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa05_split import apply_split_to_dataset, load_subject_split
from etapa06_normalize import normalize_loso_fold
from etapa07_windows import create_windows_from_dataset
from etapa11_loss_metrics import compute_regression_metrics
from etapa13_loso import load_dev_dataset


def extract_window_features(X: np.ndarray) -> np.ndarray:
    """
    Features por janela: stats das 6 colunas IMU + acc_mag + gyro_mag.
    X shape: (n_windows, n_channels, window_size)
    """
    n_win, n_ch, _ = X.shape
    feats: list[np.ndarray] = []

    for c in range(n_ch):
        ch = X[:, c, :]
        feats.extend([
            np.mean(ch, axis=1),
            np.std(ch, axis=1),
            np.min(ch, axis=1),
            np.max(ch, axis=1),
            np.max(ch, axis=1) - np.min(ch, axis=1),
            np.sqrt(np.mean(ch ** 2, axis=1)),
        ])

    if n_ch >= 6:
        acc = X[:, :3, :]
        gyro = X[:, 3:6, :]
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
        for mag in (acc_mag, gyro_mag):
            feats.extend([
                np.mean(mag, axis=1),
                np.std(mag, axis=1),
                np.min(mag, axis=1),
                np.max(mag, axis=1),
                np.max(mag, axis=1) - np.min(mag, axis=1),
                np.sqrt(np.mean(mag ** 2, axis=1)),
            ])

    return np.column_stack(feats)


def _predict_baseline(name: str, y_train: np.ndarray, X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    if name == "mean":
        return np.full(len(X_val), float(np.mean(y_train)))
    if name == "median":
        return np.full(len(X_val), float(np.median(y_train)))
    if name == "ridge":
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)
        model = Ridge(alpha=1.0)
        model.fit(Xt, y_train)
        return model.predict(Xv)
    if name == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model.predict(X_val)
    raise ValueError(name)


def run_baselines(config: ExperimentConfig | None = None) -> dict[str, Path]:
    if config is None:
        config = build_default_config()
    if config.target_mode == "curve":
        print("Baselines omitidos no modo curve (use comparação com DL de curva).\n")
        return {}
    paths = create_output_dirs(config.output_dir)
    dev_dataset = load_dev_dataset(config)

    baseline_names = ("mean", "median", "ridge", "random_forest")
    per_fold_rows: list[dict] = []

    for val_sid in dev_dataset.get_subject_ids():
        fold = normalize_loso_fold(dev_dataset, val_subject_id=val_sid)
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

        X_tr = extract_window_features(train_batch.X)
        X_va = extract_window_features(val_batch.X)
        y_tr_scaled = train_batch.y.astype(float)
        y_va_scaled = val_batch.y.astype(float)

        # Métricas em escala normalizada (comparável ao LOSO DL)
        for bname in baseline_names:
            y_pred = _predict_baseline(bname, y_tr_scaled, X_tr, X_va)
            m = compute_regression_metrics(y_va_scaled, y_pred)
            per_fold_rows.append(
                {
                    "val_subject_id": val_sid,
                    "baseline": bname,
                    "mae": m.mae,
                    "rmse": m.rmse,
                    "r2": m.r2,
                    "bias": m.bias,
                    "n_val_windows": len(y_va_scaled),
                }
            )

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_path = paths["metrics"] / "baseline_results_per_fold.csv"
    per_fold_df.to_csv(per_fold_path, index=False)

    summary = (
        per_fold_df.groupby("baseline")
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"))
        .reset_index()
    )
    summary_path = paths["metrics"] / "baseline_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Comparação DL vs baselines (usa LOSO etapa13 se existir)
    loso_path = paths["metrics"] / "etapa13_loso_summary.json"
    comp_rows = []
    if loso_path.is_file():
        import json

        loso = json.loads(loso_path.read_text(encoding="utf-8"))["aggregate"]
        comp_rows.append({"method": "deep_learning_loso", "mae_mean": loso["mae_mean"], "rmse_mean": loso["rmse_mean"]})
    for _, row in summary.iterrows():
        comp_rows.append({"method": row["baseline"], "mae_mean": row["mae_mean"], "rmse_mean": row["rmse_mean"]})
    comp_path = paths["metrics"] / "comparison_dl_vs_baselines.csv"
    pd.DataFrame(comp_rows).to_csv(comp_path, index=False)

    print("Baselines concluídos:")
    print(f"  {per_fold_path}")
    print(f"  {summary_path}")
    print(f"  {comp_path}")
    return {"per_fold": per_fold_path, "summary": summary_path, "comparison": comp_path}
