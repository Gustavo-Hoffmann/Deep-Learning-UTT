#!/usr/bin/env python3
"""
Utilitários de análise: viés proporcional, score composto, métricas em cm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from etapa01_setup import ExperimentConfig
from etapa11_loss_metrics import RegressionMetrics, TrainingConfig, compute_regression_metrics


@dataclass
class ProportionalBiasResult:
    slope: float
    intercept: float
    r2: float
    p_value: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "r2": self.r2,
            "p_value": self.p_value,
        }


def collect_oof_predictions_cm(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    dev_dataset,
    paths: dict[str, Path],
    artifact_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Reconstrói predições OOF em cm a partir dos checkpoints LOSO salvos."""
    from etapa01_setup import get_device
    from etapa11_loss_metrics import build_criterion, evaluate_loader, metrics_to_original_scale
    from etapa12_train_fold import build_model_from_config, prepare_loso_fold
    import torch

    device = get_device()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    subject_all: list[np.ndarray] = []
    best_epochs: list[int] = []

    for val_sid in dev_dataset.get_subject_ids():
        ckpt = paths["checkpoints"] / f"{artifact_prefix}_loso_val_{val_sid}_best.pt"
        metrics_path = paths["metrics"] / f"{artifact_prefix}_loso_val_{val_sid}_metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_epochs.append(int(payload.get("best_epoch", 0)))
        if not ckpt.is_file():
            continue

        loaders, fold = prepare_loso_fold(config, train_cfg, val_subject_id=val_sid, dev_dataset=dev_dataset)
        model = build_model_from_config(config, train_cfg).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        criterion = build_criterion(train_cfg)
        _, _, y_true, y_pred = evaluate_loader(model, loaders.val, criterion, device)
        yt_cm, yp_cm = metrics_to_original_scale(y_true, y_pred, fold.bundle)
        y_true_all.append(yt_cm)
        y_pred_all.append(yp_cm)
        subject_all.append(np.full(len(yt_cm), val_sid, dtype=object))

    if not y_true_all:
        return np.array([]), np.array([]), np.array([]), best_epochs
    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(subject_all),
        best_epochs,
    )


def proportional_bias_analysis(y_true_cm: np.ndarray, y_pred_cm: np.ndarray) -> ProportionalBiasResult:
    """
    Regressão: diferença ~ média, onde diferença = pred - real, média = (pred+real)/2.
    """
    yt = np.asarray(y_true_cm, dtype=float).ravel()
    yp = np.asarray(y_pred_cm, dtype=float).ravel()
    diff = yp - yt
    mean = (yp + yt) / 2.0

    if len(yt) < 2:
        return ProportionalBiasResult(0.0, float(np.mean(diff)), float("nan"), None)

    X = mean.reshape(-1, 1)
    reg = LinearRegression().fit(X, diff)
    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(reg.score(X, diff))

    p_value: float | None = None
    try:
        import statsmodels.api as sm

        X_sm = sm.add_constant(mean)
        model = sm.OLS(diff, X_sm).fit()
        p_value = float(model.pvalues[1]) if len(model.pvalues) > 1 else None
    except ImportError:
        pass

    return ProportionalBiasResult(slope=slope, intercept=intercept, r2=r2, p_value=p_value)


def composite_score(
    mae_mean: float,
    mae_std: float,
    rmse_mean: float,
    bias: float,
    proportional_bias_slope: float,
    seed_stability_penalty: float = 0.0,
) -> float:
    """Score menor = melhor candidato."""
    return (
        mae_mean
        + 0.25 * mae_std
        + 0.15 * abs(bias)
        + 0.10 * abs(proportional_bias_slope)
        + 0.10 * rmse_mean
        + seed_stability_penalty
    )


def metrics_from_cm_arrays(y_true_cm: np.ndarray, y_pred_cm: np.ndarray) -> RegressionMetrics:
    return compute_regression_metrics(y_true_cm, y_pred_cm)


def aggregate_subject_metrics(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    y_true_col: str = "y_true_cm",
    y_pred_col: str = "y_pred_cm",
    agg: str = "mean",
) -> pd.DataFrame:
    """Agrega predições por sujeito (média ou mediana das janelas)."""
    rows: list[dict[str, Any]] = []
    for sid, grp in df.groupby(subject_col):
        if agg == "median":
            yt = float(np.median(grp[y_true_col]))
            yp = float(np.median(grp[y_pred_col]))
        else:
            yt = float(np.mean(grp[y_true_col]))
            yp = float(np.mean(grp[y_pred_col]))
        m = compute_regression_metrics(np.array([yt]), np.array([yp]))
        rows.append(
            {
                "subject_id": sid,
                "y_true_cm": yt,
                "y_pred_cm": yp,
                "mae_cm": abs(yp - yt),
                "error_cm": yp - yt,
                "rmse_cm": m.rmse,
                "bias_cm": m.bias,
            }
        )
    return pd.DataFrame(rows)


def fit_linear_calibration(y_true_cm: np.ndarray, y_pred_cm: np.ndarray) -> tuple[float, float]:
    """Ajusta y_real = a + b * y_pred (apenas em predições LOSO OOF)."""
    yt = np.asarray(y_true_cm, dtype=float).ravel()
    yp = np.asarray(y_pred_cm, dtype=float).ravel()
    X = np.column_stack([np.ones(len(yp)), yp])
    coef, _, _, _ = np.linalg.lstsq(X, yt, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_calibration(y_pred_cm: np.ndarray, a: float, b: float) -> np.ndarray:
    return a + b * np.asarray(y_pred_cm, dtype=float)
