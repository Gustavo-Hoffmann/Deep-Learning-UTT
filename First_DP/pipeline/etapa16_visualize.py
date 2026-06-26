#!/usr/bin/env python3
"""
Etapa 16 — Visualização dos resultados
======================================
Gera gráficos de interpretação a partir das predições salvas.

Fontes de dados
---------------
  test (padrão) : outputs/.../predictions/etapa15_test_predictions.csv
  loso          : outputs/.../metrics/etapa13_loso_per_subject.csv (só MAE por sujeito)

NÃO executa automaticamente — rode após a Etapa 15 (ou parcialmente após Etapa 13).

Como rodar no futuro
--------------------
    cd /Users/Rodacki/Desktop/Hoffmann/UTT

    # verificar arquivos disponíveis
    .venv/bin/python pipeline/etapa16_visualize.py --list

    # gráficos do teste final (Etapa 15)
    .venv/bin/python pipeline/etapa16_visualize.py --run --source test

    # gráfico resumo LOSO dev (Etapa 13)
    .venv/bin/python pipeline/etapa16_visualize.py --run --source loso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etapa01_setup import build_default_config, create_output_dirs


PLOT_INTERPRETATIONS: dict[str, str] = {
    "true_vs_pred": (
        "Cada ponto = uma janela. Proximidade à diagonal y=x indica boa predição. "
        "Dispersão lateral = erro aleatório; curvatura = viés sistemático."
    ),
    "error_by_subject": (
        "MAE por sujeito de teste. Identifica indivíduos onde o modelo generaliza mal. "
        "Outliers aqui merecem inspeção clínica e de qualidade do sinal."
    ),
    "residuals": (
        "Distribuição dos resíduos (pred − real). Ideal: centrada em zero, simétrica. "
        "Caudas pesadas = erros grandes ocasionais; deslocamento = bias."
    ),
    "bland_altman": (
        "Compara média (real+pred)/2 com diferença (pred−real). Mostra bias constante "
        "e se o erro cresce com a amplitude (heterocedasticidade)."
    ),
    "error_by_amplitude": (
        "MAE médio por faixa de amplitude real. Revela se o modelo erra mais em "
        "movimentos pequenos ou grandes — relevante para validade clínica."
    ),
    "timeline_examples": (
        "Série temporal de amplitudes preditas vs reais ao longo do tempo, por sujeito. "
        "Mostra se o modelo acompanha tendências ou só a média."
    ),
    "loso_per_subject": (
        "MAE de cada fold LOSO no grupo dev. Estima generalização interna antes do teste final."
    ),
}


# ---------------------------------------------------------------------------
# 1. Carregar dados
# ---------------------------------------------------------------------------
def predictions_path(output_dir: Path) -> Path:
    return output_dir / "predictions" / "etapa15_test_predictions.csv"


def loso_metrics_path(output_dir: Path) -> Path:
    return output_dir / "metrics" / "etapa13_loso_per_subject.csv"


def load_test_predictions(output_dir: Path) -> pd.DataFrame:
    path = predictions_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Predições de teste não encontradas: {path}\n"
            "Execute primeiro: .venv/bin/python pipeline/etapa15_final_test.py --run"
        )
    df = pd.read_csv(path)
    required = {"subject_id", "y_true_cm", "y_pred_cm", "error_cm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")
    return df


def load_loso_per_subject(output_dir: Path) -> pd.DataFrame:
    path = loso_metrics_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Métricas LOSO não encontradas: {path}\n"
            "Execute primeiro: .venv/bin/python pipeline/etapa13_loso.py"
        )
    return pd.read_csv(path)


def check_available_sources(output_dir: Path) -> dict[str, bool]:
    return {
        "test": predictions_path(output_dir).is_file(),
        "loso": loso_metrics_path(output_dir).is_file(),
        "test_metrics": (output_dir / "metrics" / "etapa15_test_metrics.json").is_file(),
    }


# ---------------------------------------------------------------------------
# 2. Gráficos — teste final (Etapa 15)
# ---------------------------------------------------------------------------
def plot_true_vs_pred(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["y_true_cm"], df["y_pred_cm"], alpha=0.5, s=20, edgecolors="none")
    lims = [
        min(df["y_true_cm"].min(), df["y_pred_cm"].min()),
        max(df["y_true_cm"].max(), df["y_pred_cm"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Amplitude real Vicon (cm)")
    ax.set_ylabel("Amplitude predita (cm)")
    ax.set_title("Real vs predito — teste final")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_subject(df: pd.DataFrame, out_path: Path) -> None:
    err = df.groupby("subject_id")["error_cm"].apply(lambda x: np.mean(np.abs(x))).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    err.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xlabel("Sujeito (teste)")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("Erro médio absoluto por sujeito — teste final")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distribution(df: pd.DataFrame, out_path: Path) -> None:
    residuals = df["error_cm"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(float(np.mean(residuals)), color="red", linestyle="-", linewidth=1, label=f"média={np.mean(residuals):.3f}")
    ax.set_xlabel("Resíduo (pred − real) [cm]")
    ax.set_ylabel("Contagem")
    ax.set_title("Distribuição dos resíduos — teste final")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_bland_altman(df: pd.DataFrame, out_path: Path) -> None:
    y_true = df["y_true_cm"].to_numpy()
    y_pred = df["y_pred_cm"].to_numpy()
    mean = (y_true + y_pred) / 2.0
    diff = y_pred - y_true
    md = float(np.mean(diff))
    sd = float(np.std(diff))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean, diff, alpha=0.5, s=20, edgecolors="none")
    ax.axhline(md, color="red", linestyle="-", label=f"bias={md:.3f} cm")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle="--", label=f"+1.96 SD")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle="--", label=f"−1.96 SD")
    ax.set_xlabel("Média (real + pred) / 2 [cm]")
    ax.set_ylabel("Diferença (pred − real) [cm]")
    ax.set_title("Bland-Altman — teste final")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_amplitude(df: pd.DataFrame, out_path: Path, n_bins: int = 8) -> None:
    df = df.copy()
    df["abs_error"] = df["error_cm"].abs()
    df["amp_bin"] = pd.qcut(df["y_true_cm"], q=min(n_bins, df["y_true_cm"].nunique()), duplicates="drop")
    grouped = df.groupby("amp_bin", observed=True)["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    grouped.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_xlabel("Faixa de amplitude real (cm)")
    ax.set_ylabel("MAE (cm)")
    ax.set_title("Erro por faixa de amplitude — teste final")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_timeline_examples(
    df: pd.DataFrame,
    out_path: Path,
    max_subjects: int = 4,
) -> None:
    """Amplitude real vs predita ao longo do tempo (por sujeito de teste)."""
    subjects = sorted(df["subject_id"].unique())[:max_subjects]
    n = len(subjects)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sid in zip(axes, subjects):
        sub = df[df["subject_id"] == sid].sort_values("window_start_time")
        t = sub["window_start_time"].to_numpy()
        ax.plot(t, sub["y_true_cm"], label="real", color="darkorange", linewidth=1)
        ax.plot(t, sub["y_pred_cm"], label="predito", color="steelblue", linewidth=1, alpha=0.85)
        ax.set_ylabel("cm")
        ax.set_title(f"Sujeito {sid}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle("Exemplos temporais — amplitude por janela (teste final)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Gráficos — LOSO dev (Etapa 13)
# ---------------------------------------------------------------------------
def plot_loso_mae_per_subject(df: pd.DataFrame, out_path: Path) -> None:
    df = df.sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["val_subject_id"].astype(str), df["mae"], color="slateblue")
    ax.set_xlabel("Sujeito (validação LOSO)")
    ax.set_ylabel("MAE (escala normalizada)")
    ax.set_title("LOSO dev — MAE por sujeito (Etapa 13)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Orquestração
# ---------------------------------------------------------------------------
def run_test_visualizations(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    saved: list[Path] = []
    specs = [
        ("etapa16_test_true_vs_pred.png", plot_true_vs_pred),
        ("etapa16_test_error_by_subject.png", plot_error_by_subject),
        ("etapa16_test_residuals.png", plot_residual_distribution),
        ("etapa16_test_bland_altman.png", plot_bland_altman),
        ("etapa16_test_error_by_amplitude.png", plot_error_by_amplitude),
        ("etapa16_test_timeline_examples.png", plot_timeline_examples),
    ]
    for name, fn in specs:
        path = plots_dir / name
        fn(df, path)
        saved.append(path)
    return saved


def run_loso_visualizations(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    path = plots_dir / "etapa16_loso_mae_per_subject.png"
    plot_loso_mae_per_subject(df, path)
    return [path]


def print_list_report(available: dict[str, bool], output_dir: Path) -> None:
    print("=" * 60)
    print("ETAPA 16 — Visualização dos resultados (prévia)")
    print("=" * 60)
    print(f"Pasta de saída: {output_dir / 'plots'}\n")
    print("Arquivos disponíveis:")
    print(f"  teste (Etapa 15) : {'✓' if available['test'] else '✗'}  {predictions_path(output_dir)}")
    print(f"  LOSO (Etapa 13)  : {'✓' if available['loso'] else '✗'}  {loso_metrics_path(output_dir)}")
    print(f"  métricas teste   : {'✓' if available['test_metrics'] else '✗'}")

    print("\nGráficos gerados com --source test (requer Etapa 15):")
    for key in ["true_vs_pred", "error_by_subject", "residuals", "bland_altman", "error_by_amplitude", "timeline_examples"]:
        print(f"  • {key}")
        print(f"    {PLOT_INTERPRETATIONS[key]}")

    print("\nGráficos gerados com --source loso (requer Etapa 13):")
    print(f"  • loso_per_subject")
    print(f"    {PLOT_INTERPRETATIONS['loso_per_subject']}")

    print("\nComandos:")
    if available["test"]:
        print("  .venv/bin/python pipeline/etapa16_visualize.py --run --source test")
    else:
        print("  (teste) rode Etapa 15 primeiro")
    if available["loso"]:
        print("  .venv/bin/python pipeline/etapa16_visualize.py --run --source loso")
    print("=" * 60)


def print_run_report(saved: list[Path], source: str) -> None:
    print("=" * 60)
    print(f"ETAPA 16 — Gráficos gerados ({source})")
    print("=" * 60)
    for p in saved:
        print(f"  {p}")
    print("\nComo interpretar: veja PLOT_INTERPRETATIONS no topo de etapa16_visualize.py")
    print("Próxima etapa: comparação com baselines (Etapa 17).")
    print("=" * 60)


def run_stage16_visualize(source: str = "test") -> list[Path]:
    config = build_default_config()
    paths = create_output_dirs(config.output_dir)
    plots_dir = paths["plots"]

    if source == "test":
        df = load_test_predictions(config.output_dir)
        saved = run_test_visualizations(df, plots_dir)
    elif source == "loso":
        df = load_loso_per_subject(config.output_dir)
        saved = run_loso_visualizations(df, plots_dir)
    elif source == "all":
        saved = []
        if predictions_path(config.output_dir).is_file():
            saved.extend(run_test_visualizations(load_test_predictions(config.output_dir), plots_dir))
        if loso_metrics_path(config.output_dir).is_file():
            saved.extend(run_loso_visualizations(load_loso_per_subject(config.output_dir), plots_dir))
        if not saved:
            raise FileNotFoundError("Nenhuma fonte de dados disponível para --source all.")
    else:
        raise ValueError(f"source desconhecido: {source!r}")

    print_run_report(saved, source)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 16 — visualização dos resultados.")
    parser.add_argument("--list", action="store_true", help="verificar dados disponíveis")
    parser.add_argument("--run", action="store_true", help="gerar gráficos")
    parser.add_argument(
        "--source",
        choices=("test", "loso", "all"),
        default="test",
        help="test=Etapa15, loso=Etapa13, all=ambos se existirem",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_default_config()
    avail = check_available_sources(cfg.output_dir)

    if args.list or not args.run:
        print_list_report(avail, cfg.output_dir)
    else:
        run_stage16_visualize(source=args.source)
