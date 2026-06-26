#!/usr/bin/env python3
"""Inferência em sequência completa, métricas e baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import (
    FileRecord,
    LinearBaseline,
    ScalerBundle,
    build_feature_matrix,
    enrich_dataframe,
    hann_weights,
    iter_windows,
)
from .models import ResidualTCN


@dataclass
class FileMetrics:
    file_name: str
    subject_id: str
    split: str
    method: str
    rmse_cm: float
    mae_cm: float
    pearson_r: float
    r2: float
    amp_error_cm: float
    amp_error_pct: float
    peak_error_cm: float
    offset_error_cm: float
    lag_samples: int
    lag_seconds: float
    n_samples: int


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _estimate_lag_samples(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    corr = signal.correlate(yt, yp, mode="full", method="fft")
    lags = signal.correlation_lags(len(yt), len(yp), mode="full")
    return int(lags[int(np.argmax(corr))])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_s: np.ndarray,
    *,
    file_name: str,
    subject_id: str,
    split: str,
    method: str,
) -> FileMetrics:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")
    pr = _pearson_r(yt, yp)

    amp_t = float(np.max(yt) - np.min(yt))
    amp_p = float(np.max(yp) - np.min(yp))
    amp_err = abs(amp_p - amp_t)
    amp_pct = 100.0 * amp_err / max(abs(amp_t), 1e-8)

    peak_err = abs(float(np.max(yp)) - float(np.max(yt)))
    offset_err = abs(float(np.mean(yp)) - float(np.mean(yt)))

    lag = _estimate_lag_samples(yt, yp)
    dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 1 / 60.0
    lag_sec = lag * dt

    return FileMetrics(
        file_name=file_name,
        subject_id=subject_id,
        split=split,
        method=method,
        rmse_cm=rmse,
        mae_cm=mae,
        pearson_r=pr,
        r2=r2,
        amp_error_cm=amp_err,
        amp_error_pct=amp_pct,
        peak_error_cm=peak_err,
        offset_error_cm=offset_err,
        lag_samples=lag,
        lag_seconds=lag_sec,
        n_samples=len(yt),
    )


@torch.no_grad()
def predict_full_sequence(
    model: ResidualTCN,
    df: pd.DataFrame,
    baseline: LinearBaseline,
    scalers: ScalerBundle,
    *,
    device: torch.device,
    window_size: int = 512,
    stride: int = 128,
) -> dict[str, np.ndarray]:
    enriched = enrich_dataframe(df, baseline)
    x_full = build_feature_matrix(enriched)
    T = x_full.shape[-1]

    vicon = enriched["vicon_esternoZ_cm"].to_numpy(dtype=np.float32)
    smart_disp = enriched["smart_disp_cm"].to_numpy(dtype=np.float32)
    calibrado = enriched["smart_calibrado_cm"].to_numpy(dtype=np.float32)
    time_s = enriched["Time"].to_numpy(dtype=np.float32)

    if T < window_size:
        x_scaled = scalers.transform_features(x_full)
        xt = torch.from_numpy(x_scaled).unsqueeze(0).to(device)
        resid_scaled = model(xt).squeeze(0).cpu().numpy()
        resid = scalers.inverse_residual(resid_scaled)
        pred_dl = calibrado + resid[:T]
    else:
        weights = hann_weights(window_size)
        acc_resid = np.zeros(T, dtype=np.float32)
        acc_w = np.zeros(T, dtype=np.float32)

        for start, xw in iter_windows(x_full, window_size, stride):
            x_scaled = scalers.transform_features(xw)
            xt = torch.from_numpy(x_scaled).unsqueeze(0).to(device)
            resid_scaled = model(xt).squeeze(0).cpu().numpy()
            resid = scalers.inverse_residual(resid_scaled)
            end = start + window_size
            acc_resid[start:end] += resid * weights
            acc_w[start:end] += weights

        resid_full = acc_resid / np.maximum(acc_w, 1e-8)
        pred_dl = calibrado + resid_full

    resid_pred = pred_dl - calibrado
    erro = pred_dl - vicon

    return {
        "Time": time_s,
        "vicon_cm": vicon,
        "smart_disp_cm": smart_disp,
        "smart_calibrado_cm": calibrado,
        "pred_dl_cm": pred_dl.astype(np.float32),
        "residuo_predito_cm": resid_pred.astype(np.float32),
        "erro_dl_cm": erro.astype(np.float32),
    }


def evaluate_all_files(
    model: ResidualTCN,
    files: list[FileRecord],
    baseline: LinearBaseline,
    scalers: ScalerBundle,
    *,
    device: torch.device,
    split_label: str,
    predictions_dir: Path,
    window_size: int = 512,
    stride: int = 128,
) -> tuple[list[FileMetrics], list[pd.DataFrame]]:
    model.eval()
    all_metrics: list[FileMetrics] = []
    pred_dfs: list[pd.DataFrame] = []

    for rec in files:
        pred = predict_full_sequence(
            model,
            rec.df,
            baseline,
            scalers,
            device=device,
            window_size=window_size,
            stride=stride,
        )
        time_s = pred["Time"]
        vicon = pred["vicon_cm"]
        smart = pred["smart_disp_cm"]
        cal = pred["smart_calibrado_cm"]
        dl = pred["pred_dl_cm"]

        for method, y_pred in [
            ("raw_smart", smart),
            ("linear_baseline", cal),
            ("dl_residual", dl),
        ]:
            all_metrics.append(
                compute_metrics(
                    vicon,
                    y_pred,
                    time_s,
                    file_name=rec.file_name,
                    subject_id=rec.subject_id,
                    split=split_label,
                    method=method,
                )
            )

        out_df = pd.DataFrame(
            {
                "Time": time_s,
                "subject_id": rec.subject_id,
                "file_name": rec.file_name,
                "vicon_cm": vicon,
                "smart_disp_cm": smart,
                "smart_calibrado_cm": cal,
                "pred_dl_cm": dl,
                "residuo_predito_cm": pred["residuo_predito_cm"],
                "erro_dl_cm": pred["erro_dl_cm"],
                "split": split_label,
            }
        )
        pred_dfs.append(out_df)
        out_path = predictions_dir / f"{rec.subject_id}_{rec.file_name.replace('.csv', '')}_pred.csv"
        out_df.to_csv(out_path, index=False)

    return all_metrics, pred_dfs


def metrics_to_dataframe(metrics: list[FileMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in metrics])


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, grp in metrics_df.groupby("method"):
        rows.append(
            {
                "method": method,
                "n_files": len(grp),
                "rmse_cm_mean": grp["rmse_cm"].mean(),
                "rmse_cm_std": grp["rmse_cm"].std(),
                "mae_cm_mean": grp["mae_cm"].mean(),
                "pearson_r_mean": grp["pearson_r"].mean(),
                "r2_mean": grp["r2"].mean(),
                "amp_error_cm_mean": grp["amp_error_cm"].mean(),
                "offset_error_cm_mean": grp["offset_error_cm"].mean(),
                "lag_seconds_mean": grp["lag_seconds"].mean(),
            }
        )
    summary = pd.DataFrame(rows)

    if {"linear_baseline", "dl_residual"}.issubset(set(summary["method"])):
        lin = summary.loc[summary["method"] == "linear_baseline", "rmse_cm_mean"].iloc[0]
        dl = summary.loc[summary["method"] == "dl_residual", "rmse_cm_mean"].iloc[0]
        improvement = 100.0 * (lin - dl) / max(lin, 1e-8)
        summary.loc[summary["method"] == "dl_residual", "improvement_vs_linear_pct"] = improvement

    return summary


def add_improvement_column(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona melhora percentual da DL vs linear por arquivo."""
    out_rows = []
    for file_name, grp in metrics_df.groupby("file_name"):
        lin = grp.loc[grp["method"] == "linear_baseline", "rmse_cm"]
        dl = grp.loc[grp["method"] == "dl_residual", "rmse_cm"]
        imp = None
        if len(lin) and len(dl):
            imp = 100.0 * (float(lin.iloc[0]) - float(dl.iloc[0])) / max(float(lin.iloc[0]), 1e-8)
        for _, row in grp.iterrows():
            d = row.to_dict()
            if row["method"] == "dl_residual":
                d["improvement_vs_linear_pct"] = imp
            out_rows.append(d)
    return pd.DataFrame(out_rows)
