#!/usr/bin/env python3
"""Avaliação detalhada, Bland-Altman, calibração e métricas por janela/sujeito."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    aggregate_subject_metrics,
    apply_calibration,
    fit_linear_calibration,
    proportional_bias_analysis,
)
from curve_utils import reconstruct_subject_curve
from etapa01_setup import build_default_config, create_output_dirs
from etapa11_loss_metrics import compute_regression_metrics


def plot_bland_altman(
    y_true_cm: np.ndarray,
    y_pred_cm: np.ndarray,
    output_path: Path,
    title: str,
    subject_ids: np.ndarray | None = None,
) -> Path:
    diff = y_pred_cm - y_true_cm
    mean = (y_pred_cm + y_true_cm) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    loa_hi = bias + 1.96 * sd
    loa_lo = bias - 1.96 * sd

    pb = proportional_bias_analysis(y_true_cm, y_pred_cm)
    x_line = np.linspace(mean.min(), mean.max(), 50)
    trend = pb.intercept + pb.slope * x_line

    fig, ax = plt.subplots(figsize=(8, 5))
    if subject_ids is not None:
        uniq = sorted(set(subject_ids))
        cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        color_map = {s: cmap[i % len(cmap)] for i, s in enumerate(uniq)}
        for s in uniq:
            mask = subject_ids == s
            ax.scatter(mean[mask], diff[mask], s=12, alpha=0.6, color=color_map[s], label=s)
    else:
        ax.scatter(mean, diff, s=12, alpha=0.5, color="steelblue")

    ax.axhline(bias, color="red", linestyle="-", label=f"bias={bias:.3f}")
    ax.axhline(loa_hi, color="gray", linestyle="--", label=f"+1.96SD={loa_hi:.3f}")
    ax.axhline(loa_lo, color="gray", linestyle="--", label=f"-1.96SD={loa_lo:.3f}")
    ax.plot(x_line, trend, color="darkorange", linestyle="-", label=f"tendência (slope={pb.slope:.4f})")
    ax.set_xlabel("Média (real + pred) / 2 [cm]")
    ax.set_ylabel("Diferença (pred - real) [cm]")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_curve_overlay(
    pred_df: pd.DataFrame,
    output_path: Path,
    *,
    max_subjects: int = 4,
    title: str = "Vicon real vs predito (IMU)",
) -> Path:
    """Sobrepõe curvas reconstruídas por sujeito."""
    subjects = sorted(pred_df["subject_id"].astype(str).unique())[:max_subjects]
    n = len(subjects)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, subjects):
        curve = reconstruct_subject_curve(pred_df, sid)
        ax.plot(curve["time_s"], curve["y_true_cm"], label="Vicon real", color="darkorange", linewidth=1.2)
        ax.plot(curve["time_s"], curve["y_pred_cm"], label="IMU predito", color="steelblue", linewidth=1.0, alpha=0.9)
        ax.set_ylabel("cm")
        ax.set_title(f"Sujeito {sid}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_window_and_subject_metrics(
    pred_df: pd.DataFrame,
    metrics_dir: Path,
    plots_dir: Path,
    prefix: str = "eval",
    *,
    skip_bland_altman: bool = False,
    plot_curve_overlay_flag: bool = True,
) -> dict[str, Path]:
    """Salva métricas e gráficos por janela e por sujeito."""
    paths: dict[str, Path] = {}

    win_m = compute_regression_metrics(pred_df["y_true_cm"], pred_df["y_pred_cm"])
    win_row = {"level": "window", **win_m.to_dict()}
    paths["window"] = metrics_dir / f"{prefix}_window_level_metrics.csv"
    pd.DataFrame([win_row]).to_csv(paths["window"], index=False)

    subj_mean = aggregate_subject_metrics(pred_df, agg="mean")
    subj_med = aggregate_subject_metrics(pred_df, agg="median")
    paths["subject_mean"] = metrics_dir / "subject_level_mean_metrics.csv"
    paths["subject_median"] = metrics_dir / "subject_level_median_metrics.csv"
    subj_mean.to_csv(paths["subject_mean"], index=False)
    subj_med.to_csv(paths["subject_median"], index=False)

    # Gráficos
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_df["y_true_cm"], pred_df["y_pred_cm"], alpha=0.4, s=10)
    lim = [min(pred_df["y_true_cm"].min(), pred_df["y_pred_cm"].min()),
           max(pred_df["y_true_cm"].max(), pred_df["y_pred_cm"].max())]
    ax.plot(lim, lim, "k--", alpha=0.5)
    ax.set_xlabel("Real (cm)")
    ax.set_ylabel("Predito (cm)")
    ax.set_title(f"Real vs Predito — {prefix}")
    fig.tight_layout()
    p = plots_dir / f"{prefix}_true_vs_pred_window.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)

    if plot_curve_overlay_flag and "time_s" in pred_df.columns:
        plot_curve_overlay(
            pred_df,
            plots_dir / f"{prefix}_curve_overlay.png",
            title=f"Vicon vs predito — {prefix}",
        )

    if not skip_bland_altman:
        plot_bland_altman(
            pred_df["y_true_cm"].to_numpy(),
            pred_df["y_pred_cm"].to_numpy(),
            plots_dir / f"{prefix}_bland_altman_window.png",
            f"Bland-Altman por janela — {prefix}",
            subject_ids=pred_df["subject_id"].to_numpy() if "subject_id" in pred_df.columns else None,
        )

        subj_mean_df = subj_mean
        plot_bland_altman(
            subj_mean_df["y_true_cm"].to_numpy(),
            subj_mean_df["y_pred_cm"].to_numpy(),
            plots_dir / f"{prefix}_bland_altman_subject.png",
            f"Bland-Altman por sujeito (média) — {prefix}",
        )

    return paths


def run_calibration_from_loso_oof(
    config_output_dir: Path,
    loso_prefix: str = "etapa13",
) -> tuple[float, float, Path | None]:
    """
    Ajusta calibração linear apenas com predições OOF do LOSO nos 70%.
    Retorna (a, b, path das predições OOF se existir).
    """
    paths = create_output_dirs(config_output_dir)
    oof_path = paths["predictions"] / f"{loso_prefix}_oof_predictions.csv"
    if not oof_path.is_file():
        return 0.0, 1.0, None

    df = pd.read_csv(oof_path)
    a, b = fit_linear_calibration(df["y_true_cm"].to_numpy(), df["y_pred_cm"].to_numpy())
    cal_payload = {"a": a, "b": b, "formula": "y_real_cm = a + b * y_pred_cm"}
    cal_path = paths["configs"] / "loso_calibration.json"
    cal_path.write_text(json.dumps(cal_payload, indent=2), encoding="utf-8")
    return a, b, oof_path


def apply_test_calibration(
    test_pred_path: Path,
    a: float,
    b: float,
    metrics_dir: Path,
) -> dict:
    df = pd.read_csv(test_pred_path)
    df["y_pred_cm_calibrated"] = apply_calibration(df["y_pred_cm"].to_numpy(), a, b)
    m_before = compute_regression_metrics(df["y_true_cm"], df["y_pred_cm"])
    m_after = compute_regression_metrics(df["y_true_cm"], df["y_pred_cm_calibrated"])
    payload = {
        "before_calibration": m_before.to_dict(),
        "after_calibration": m_after.to_dict(),
        "calibration": {"a": a, "b": b},
    }
    out = metrics_dir / "test_metrics_with_calibration.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
