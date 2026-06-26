#!/usr/bin/env python3
"""Diagnósticos de dados antes do treino (--diagnostics)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa03_load import load_all_subjects
from etapa05_split import apply_split_to_dataset, load_subject_split


def _y_cm_per_subject(record) -> dict:
    vicon = record.dataframe["vicon"].to_numpy(dtype=float)
    return {
        "subject_id": record.subject_id,
        "n_samples": len(vicon),
        "duration_s": float(record.duration_s or len(vicon) / 60.0),
        "y_cm_mean": float(np.mean(vicon)),
        "y_cm_std": float(np.std(vicon)),
        "y_cm_min": float(np.min(vicon)),
        "y_cm_max": float(np.max(vicon)),
        "y_cm_range": float(np.max(vicon) - np.min(vicon)),
        "amplitude_cm": float(np.max(vicon) - np.min(vicon)),
    }


def run_diagnostics(config: ExperimentConfig | None = None, output_subdir: str = "diagnostics") -> Path:
    if config is None:
        config = build_default_config()
    paths = create_output_dirs(config.output_dir)
    out_dir = paths["root"] / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = paths["plots"]
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_all_subjects(config.data_dir)
    split = load_subject_split(paths["splits"])
    dev_ds, test_ds = apply_split_to_dataset(dataset, split)

    rows = []
    for rec in dev_ds.subjects.values():
        row = _y_cm_per_subject(rec)
        row["split"] = "dev"
        rows.append(row)
    for rec in test_ds.subjects.values():
        row = _y_cm_per_subject(rec)
        row["split"] = "test"
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = paths["metrics"] / "diagnostics_subjects.csv"
    df.to_csv(csv_path, index=False)
    df.to_csv(out_dir / "diagnostics_subjects.csv", index=False)

    # Histograma geral de amplitude (range por sujeito)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["amplitude_cm"], bins=15, color="steelblue", edgecolor="white")
    ax.set_xlabel("Amplitude Vicon (cm) por sujeito")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuição de amplitude (max-min) por sujeito")
    fig.tight_layout()
    p1 = plots_dir / "diagnostics_y_distribution.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Boxplot por sujeito
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.4), 5))
    dev = df[df["split"] == "dev"]
    test = df[df["split"] == "test"]
    bp = ax.boxplot([dev["amplitude_cm"], test["amplitude_cm"]])
    ax.set_xticklabels(["dev (70%)", "test (30%)"])
    ax.set_ylabel("Amplitude (cm)")
    ax.set_title("Amplitude por grupo de split")
    fig.tight_layout()
    p2 = plots_dir / "diagnostics_dev_vs_test.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Por sujeito
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue" if s == "dev" else "coral" for s in df["split"]]
    ax.bar(df["subject_id"], df["amplitude_cm"], color=colors)
    ax.set_xlabel("Sujeito")
    ax.set_ylabel("Amplitude (cm)")
    ax.set_title("Amplitude por sujeito (azul=dev, coral=test)")
    plt.xticks(rotation=45)
    fig.tight_layout()
    p3 = plots_dir / "diagnostics_y_by_subject.png"
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Diagnósticos salvos em {out_dir}")
    print(f"  {csv_path}")
    print(f"  {p1.name}, {p2.name}, {p3.name}")
    return out_dir
