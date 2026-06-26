#!/usr/bin/env python3
"""Utilitários para modo curve: métricas, exportação e reconstrução temporal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def is_curve_array(y: np.ndarray) -> bool:
    arr = np.asarray(y)
    return arr.ndim >= 2 and arr.shape[-1] > 1


def curve_window_mae(yt: np.ndarray, yp: np.ndarray) -> float:
    """MAE médio dentro de cada janela, depois média entre janelas."""
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if yt.ndim == 1:
        return float(np.mean(np.abs(yt - yp)))
    return float(np.mean(np.mean(np.abs(yt - yp), axis=1)))


def curve_window_rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if yt.ndim == 1:
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    per = np.sqrt(np.mean((yt - yp) ** 2, axis=1))
    return float(np.mean(per))


def curve_window_rms(yt: np.ndarray, yp: np.ndarray) -> float:
    """RMS do erro por janela (média das RMS intra-janela)."""
    return curve_window_rmse(yt, yp)


def predictions_to_long_rows(
    *,
    subject_id: str,
    window_start_times: np.ndarray,
    y_true_cm: np.ndarray,
    y_pred_cm: np.ndarray,
    sampling_hz: float,
    window_samples: int,
) -> list[dict[str, Any]]:
    """Converte predições (escalar ou curva) em linhas longas com tempo absoluto."""
    yt = np.asarray(y_true_cm, dtype=float)
    yp = np.asarray(y_pred_cm, dtype=float)
    times = np.asarray(window_start_times, dtype=float).ravel()
    dt = 1.0 / sampling_hz
    rows: list[dict[str, Any]] = []

    # Uma janela com curva (T amostras) pode chegar como vetor 1D + um único tempo.
    if yt.ndim == 1 and len(times) == 1 and len(yt) > 1:
        yt = yt.reshape(1, -1)
        yp = yp.reshape(1, -1)

    if yt.ndim == 1:
        for i in range(len(yt)):
            rows.append(
                {
                    "subject_id": subject_id,
                    "window_start_time": float(times[i]),
                    "time_s": float(times[i]),
                    "sample_in_window": 0,
                    "y_true_cm": float(yt[i]),
                    "y_pred_cm": float(yp[i]),
                    "error_cm": float(yp[i] - yt[i]),
                }
            )
        return rows

    n_win = yt.shape[0]
    for i in range(n_win):
        t0 = float(times[i]) if i < len(times) else float(i) * dt
        for j in range(yt.shape[1]):
            rows.append(
                {
                    "subject_id": subject_id,
                    "window_start_time": t0,
                    "time_s": t0 + j * dt,
                    "sample_in_window": int(j),
                    "y_true_cm": float(yt[i, j]),
                    "y_pred_cm": float(yp[i, j]),
                    "error_cm": float(yp[i, j] - yt[i, j]),
                }
            )
    return rows


def reconstruct_subject_curve(
    df: pd.DataFrame,
    subject_id: str,
    *,
    time_col: str = "time_s",
    true_col: str = "y_true_cm",
    pred_col: str = "y_pred_cm",
) -> pd.DataFrame:
    """
    Reconstrói série contínua por média de janelas sobrepostas (mesmo time_s).
    """
    sub = df[df["subject_id"].astype(str) == str(subject_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[time_col, true_col, pred_col])

    if "sample_in_window" in sub.columns and sub["sample_in_window"].nunique() > 1:
        grouped = (
            sub.groupby(time_col, as_index=False)
            .agg({true_col: "mean", pred_col: "mean"})
            .sort_values(time_col)
        )
        return grouped

    return sub.sort_values(time_col)[[time_col, true_col, pred_col]].reset_index(drop=True)
