#!/usr/bin/env python3
"""Gráficos automáticos de validação e comparação de métodos."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_name(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_")


def plot_file_curves(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    t = df["Time"].to_numpy()
    axes[0].plot(t, df["vicon_cm"], label="Vicon", linewidth=2, color="black")
    axes[0].plot(t, df["smart_disp_cm"], label="Smart raw (cm)", alpha=0.7)
    axes[0].plot(t, df["smart_calibrado_cm"], label="Smart calibrado linear", alpha=0.8)
    axes[0].plot(t, df["pred_dl_cm"], label="DL residual", alpha=0.9)
    axes[0].set_ylabel("Deslocamento (cm)")
    axes[0].legend(loc="best")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, df["erro_dl_cm"], color="crimson", label="Erro DL")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Erro (cm)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_scatter_vicon_vs_dl(pred_dfs: list[pd.DataFrame], out_path: Path) -> None:
    yt, yp = [], []
    for df in pred_dfs:
        yt.extend(df["vicon_cm"].tolist())
        yp.extend(df["pred_dl_cm"].tolist())
    yt = np.asarray(yt)
    yp = np.asarray(yp)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yt, yp, alpha=0.15, s=8, edgecolors="none")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Vicon (cm)")
    ax.set_ylabel("Predição DL (cm)")
    ax.set_title("Scatter: Vicon vs DL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_bland_altman(pred_dfs: list[pd.DataFrame], out_path: Path) -> None:
    yt, yp = [], []
    for df in pred_dfs:
        yt.extend(df["vicon_cm"].tolist())
        yp.extend(df["pred_dl_cm"].tolist())
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    mean = (yt + yp) / 2
    diff = yp - yt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, diff, alpha=0.15, s=8, edgecolors="none")
    md = np.mean(diff)
    sd = np.std(diff)
    ax.axhline(md, color="k", linestyle="-", linewidth=1)
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--")
    ax.set_xlabel("Média (cm)")
    ax.set_ylabel("Diferença DL - Vicon (cm)")
    ax.set_title("Bland-Altman (ponto a ponto)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rmse_boxplot(metrics_df: pd.DataFrame, out_path: Path) -> None:
    methods = ["raw_smart", "linear_baseline", "dl_residual"]
    labels = ["Smart raw", "Linear", "DL residual"]
    data = [metrics_df.loc[metrics_df["method"] == m, "rmse_cm"].to_numpy() for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("RMSE (cm)")
    ax.set_title("RMSE por método (validação)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def generate_all_plots(
    pred_dfs: list[pd.DataFrame],
    metrics_df: pd.DataFrame,
    plots_dir: Path,
    *,
    quick: bool = False,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    for df in pred_dfs:
        fname = _safe_name(df["file_name"].iloc[0])
        sid = df["subject_id"].iloc[0]
        plot_file_curves(
            df,
            plots_dir / f"{sid}_{fname}_curves.png",
            title=f"Sujeito {sid} — {fname}",
        )

    if quick:
        plot_rmse_boxplot(metrics_df, plots_dir / "rmse_boxplot.png")
        return

    plot_scatter_vicon_vs_dl(pred_dfs, plots_dir / "scatter_vicon_vs_dl.png")
    plot_bland_altman(pred_dfs, plots_dir / "bland_altman.png")
    plot_rmse_boxplot(metrics_df, plots_dir / "rmse_boxplot.png")

    dl_metrics = metrics_df[metrics_df["method"] == "dl_residual"].copy()
    if not dl_metrics.empty:
        best = dl_metrics.nsmallest(5, "rmse_cm")
        worst = dl_metrics.nlargest(5, "rmse_cm")
        for tag, subset in [("top5_best", best), ("top5_worst", worst)]:
            for _, row in subset.iterrows():
                df = next(d for d in pred_dfs if d["file_name"].iloc[0] == row["file_name"])
                plot_file_curves(
                    df,
                    plots_dir / f"{tag}_{row['subject_id']}_{_safe_name(row['file_name'])}.png",
                    title=f"{tag} — {row['file_name']} RMSE={row['rmse_cm']:.3f}",
                )
