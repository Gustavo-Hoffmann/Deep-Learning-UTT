#!/usr/bin/env python3
"""
Etapa 4 — Conferência pós-alinhamento e qualidade dos sinais
============================================================
Assume que os dados já foram alinhados previamente (um instante = uma linha
com IMU e Vicon juntos). Verifica tempo, frequência, buracos e gera plots
simples de inspeção visual.

NÃO aplica filtros pesados. NÃO faz split. NÃO usa grupo de teste (ainda inexistente).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa02_schema import INPUT_FEATURE_COLUMNS, TIME_COLUMN, TARGET_COLUMN
from etapa03_load import LoadedDataset, SubjectRecord, load_all_subjects, run_stage03_load

# ---------------------------------------------------------------------------
# 1. Parâmetros de checagem
# ---------------------------------------------------------------------------
DEFAULT_GAP_FACTOR: float = 2.0       # dt > 2× mediana => buraco temporal
DEFAULT_TIME_TOL: float = 1e-9        # tolerância para monotonicidade
DEFAULT_FS_TOLERANCE_HZ: float = 2.0  # aviso se |fs - esperada| > tolerância


# ---------------------------------------------------------------------------
# 2. Resultados por sujeito
# ---------------------------------------------------------------------------
@dataclass
class SubjectQualityReport:
    subject_id: str
    source_file: str
    n_samples: int

    # tempo
    time_monotonic: bool
    n_duplicate_times: int
    duration_s: float
    dt_median_s: float
    dt_std_s: float
    fs_estimated_hz: float

    # buracos temporais
    n_gaps: int
    gap_indices: list[int] = field(default_factory=list)
    max_gap_s: float = 0.0

    # alinhamento IMU × Vicon (mesma tabela = mesmo nº de linhas)
    n_imu_samples: int = 0
    n_vicon_samples: int = 0
    imu_vicon_same_length: bool = True

    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.time_monotonic
            and self.n_duplicate_times == 0
            and self.imu_vicon_same_length
            and self.n_gaps == 0
        )


@dataclass
class QualitySummary:
    n_subjects: int
    n_ok: int
    n_with_warnings: int
    fs_median_hz: float
    fs_min_hz: float
    fs_max_hz: float
    subjects_with_gaps: list[str]
    subjects_non_monotonic: list[str]
    subjects_length_mismatch: list[str]
    expected_fs_hz: float | None


# ---------------------------------------------------------------------------
# 3. Funções de análise temporal
# ---------------------------------------------------------------------------
def compute_dt(time_values: np.ndarray) -> np.ndarray:
    """Intervalos entre amostras consecutivas (primeiro dt = 0)."""
    t = np.asarray(time_values, dtype=float)
    dt = np.diff(t, prepend=t[0])
    return dt


def check_time_monotonic(time_values: np.ndarray, tol: float = DEFAULT_TIME_TOL) -> tuple[bool, int]:
    """
    Verifica se o tempo é estritamente crescente.

    Retorna (monotônico, n_duplicatas).
    """
    t = np.asarray(time_values, dtype=float)
    if len(t) <= 1:
        return True, 0

    diffs = np.diff(t)
    n_duplicate = int(np.sum(diffs <= tol))
    monotonic = n_duplicate == 0
    return monotonic, n_duplicate


def estimate_sampling_rate(time_values: np.ndarray) -> tuple[float, float, float]:
    """
    Estima frequência de amostragem a partir da mediana de dt.

    Retorna (fs_hz, dt_mediana, dt_desvio).
    """
    dt = compute_dt(time_values)
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return float("nan"), float("nan"), float("nan")

    dt_med = float(np.median(positive_dt))
    dt_std = float(np.std(positive_dt))
    fs = 1.0 / dt_med if dt_med > 0 else float("nan")
    return fs, dt_med, dt_std


def find_temporal_gaps(
    time_values: np.ndarray,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> tuple[list[int], float]:
    """
    Identifica buracos temporais onde dt > gap_factor × mediana(dt).

    Retorna (índices do início do buraco, maior dt encontrado).
    """
    t = np.asarray(time_values, dtype=float)
    if len(t) <= 1:
        return [], 0.0

    dt = np.diff(t)
    positive_dt = dt[dt > 0]
    if len(positive_dt) == 0:
        return [], 0.0

    threshold = gap_factor * float(np.median(positive_dt))
    gap_idx = [int(i) for i, d in enumerate(dt) if d > threshold]
    max_gap = float(dt[gap_idx].max()) if gap_idx else 0.0
    return gap_idx, max_gap


def check_imu_vicon_alignment(df: pd.DataFrame) -> tuple[int, int, bool]:
    """
    Confere se IMU e Vicon têm o mesmo número de amostras válidas.

    Como os dados já estão alinhados em uma única tabela, esperamos o mesmo
    nº de linhas não-NaN para todas as colunas de sinal.
    """
    imu_valid = df[list(INPUT_FEATURE_COLUMNS)].notna().all(axis=1)
    vicon_valid = df[TARGET_COLUMN].notna()

    n_imu = int(imu_valid.sum())
    n_vicon = int(vicon_valid.sum())
    same_length = n_imu == n_vicon == len(df)
    return n_imu, n_vicon, same_length


# ---------------------------------------------------------------------------
# 4. Relatório de qualidade por sujeito
# ---------------------------------------------------------------------------
def analyze_subject_quality(
    record: SubjectRecord,
    expected_fs_hz: float | None = None,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    fs_tolerance_hz: float = DEFAULT_FS_TOLERANCE_HZ,
) -> SubjectQualityReport:
    """Executa todas as checagens de qualidade para um sujeito."""
    df = record.dataframe
    t = df[TIME_COLUMN].to_numpy(dtype=float)

    monotonic, n_dup = check_time_monotonic(t)
    fs, dt_med, dt_std = estimate_sampling_rate(t)
    gap_idx, max_gap = find_temporal_gaps(t, gap_factor=gap_factor)
    n_imu, n_vicon, same_len = check_imu_vicon_alignment(df)

    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    warnings: list[str] = []

    if not monotonic:
        warnings.append(f"tempo não estritamente crescente ({n_dup} duplicata(s)/retrocesso(s))")
    if gap_idx:
        warnings.append(f"{len(gap_idx)} buraco(s) temporal(is); maior dt={max_gap:.4f}s")
    if not same_len:
        warnings.append(f"IMU ({n_imu}) e Vicon ({n_vicon}) com contagem diferente")
    if expected_fs_hz is not None and not np.isnan(fs):
        if abs(fs - expected_fs_hz) > fs_tolerance_hz:
            warnings.append(
                f"fs estimada={fs:.2f} Hz difere da esperada={expected_fs_hz:.2f} Hz"
            )

    return SubjectQualityReport(
        subject_id=record.subject_id,
        source_file=record.source_file,
        n_samples=record.n_rows,
        time_monotonic=monotonic,
        n_duplicate_times=n_dup,
        duration_s=duration,
        dt_median_s=dt_med,
        dt_std_s=dt_std,
        fs_estimated_hz=fs,
        n_gaps=len(gap_idx),
        gap_indices=gap_idx,
        max_gap_s=max_gap,
        n_imu_samples=n_imu,
        n_vicon_samples=n_vicon,
        imu_vicon_same_length=same_len,
        warnings=warnings,
    )


def analyze_all_subjects(
    dataset: LoadedDataset,
    expected_fs_hz: float | None = None,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> tuple[dict[str, SubjectQualityReport], QualitySummary]:
    """Analisa qualidade de todos os sujeitos carregados."""
    reports: dict[str, SubjectQualityReport] = {}

    for sid, record in dataset.subjects.items():
        reports[sid] = analyze_subject_quality(
            record,
            expected_fs_hz=expected_fs_hz,
            gap_factor=gap_factor,
        )

    fs_values = [r.fs_estimated_hz for r in reports.values() if not np.isnan(r.fs_estimated_hz)]

    summary = QualitySummary(
        n_subjects=len(reports),
        n_ok=sum(1 for r in reports.values() if r.ok),
        n_with_warnings=sum(1 for r in reports.values() if r.warnings),
        fs_median_hz=float(np.median(fs_values)) if fs_values else float("nan"),
        fs_min_hz=float(np.min(fs_values)) if fs_values else float("nan"),
        fs_max_hz=float(np.max(fs_values)) if fs_values else float("nan"),
        subjects_with_gaps=[sid for sid, r in reports.items() if r.n_gaps > 0],
        subjects_non_monotonic=[sid for sid, r in reports.items() if not r.time_monotonic],
        subjects_length_mismatch=[
            sid for sid, r in reports.items() if not r.imu_vicon_same_length
        ],
        expected_fs_hz=expected_fs_hz,
    )

    return reports, summary


# ---------------------------------------------------------------------------
# 5. Plots de inspeção visual (sem filtros)
# ---------------------------------------------------------------------------
def plot_subject_signals(
    record: SubjectRecord,
    output_path: Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plota sinais brutos do smartphone (norma da aceleração) e Vicon vs tempo.

    Sem filtros — apenas inspeção visual do alinhamento prévio.
    """
    df = record.dataframe
    t = df[TIME_COLUMN].to_numpy(dtype=float)
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    acc_norm = np.linalg.norm(acc, axis=1)
    vicon = df[TARGET_COLUMN].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t, acc_norm, color="steelblue", linewidth=0.8, label="|aceleração|")
    axes[0].set_ylabel("m/s²")
    axes[0].set_title(f"Sujeito {record.subject_id} — smartphone (norma acc){title_suffix}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, vicon, color="darkorange", linewidth=0.8, label="Vicon (Z)")
    axes[1].set_xlabel("tempo (s)")
    axes[1].set_ylabel("cm")
    axes[1].set_title(f"Sujeito {record.subject_id} — referência Vicon{title_suffix}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")

    return fig


def plot_example_subjects(
    dataset: LoadedDataset,
    output_dir: Path,
    subject_ids: list[str] | None = None,
    max_plots: int = 3,
) -> list[Path]:
    """Gera plots para alguns sujeitos de exemplo."""
    if subject_ids is None:
        ids = dataset.get_subject_ids()
        # primeiro, mediano e último — boa variedade sem viés de teste
        picks = [ids[0], ids[len(ids) // 2], ids[-1]][:max_plots]
    else:
        picks = subject_ids[:max_plots]

    saved: list[Path] = []
    for sid in picks:
        record = dataset.subjects[sid]
        out = output_dir / f"etapa04_sinais_sujeito_{sid}.png"
        plot_subject_signals(record, output_path=out)
        plt.close()
        saved.append(out)

    return saved


# ---------------------------------------------------------------------------
# 6. Relatório legível
# ---------------------------------------------------------------------------
def print_quality_report(
    reports: dict[str, SubjectQualityReport],
    summary: QualitySummary,
) -> None:
    print("=" * 60)
    print("ETAPA 4 — Conferência pós-alinhamento e qualidade dos sinais")
    print("=" * 60)
    print(f"Sujeitos analisados : {summary.n_subjects}")
    print(f"Sem alertas         : {summary.n_ok}")
    print(f"Com alertas         : {summary.n_with_warnings}")
    print(
        f"Frequência estimada : mediana={summary.fs_median_hz:.2f} Hz "
        f"(min={summary.fs_min_hz:.2f}, max={summary.fs_max_hz:.2f})"
    )
    if summary.expected_fs_hz is not None:
        print(f"Frequência esperada : {summary.expected_fs_hz:.2f} Hz")

    if summary.subjects_non_monotonic:
        print(f"\n⚠ Tempo não monotônico: {summary.subjects_non_monotonic}")
    if summary.subjects_with_gaps:
        print(f"⚠ Buracos temporais  : {summary.subjects_with_gaps}")
    if summary.subjects_length_mismatch:
        print(f"⚠ IMU ≠ Vicon (contagem): {summary.subjects_length_mismatch}")
    if not any([
        summary.subjects_non_monotonic,
        summary.subjects_with_gaps,
        summary.subjects_length_mismatch,
    ]):
        print("\n✓ Tempo crescente, sem buracos e IMU/Vicon com mesmo nº de amostras.")

    print("\nDetalhe por sujeito:")
    print(
        f"  {'ID':<6} {'linhas':>8} {'fs(Hz)':>8} {'dt_med(s)':>10} "
        f"{'buracos':>8} {'duração':>10}  alertas"
    )
    print("  " + "-" * 72)
    for sid in sorted(reports.keys()):
        r = reports[sid]
        alert = "; ".join(r.warnings) if r.warnings else "—"
        print(
            f"  {sid:<6} {r.n_samples:>8} {r.fs_estimated_hz:>8.2f} "
            f"{r.dt_median_s:>10.6f} {r.n_gaps:>8} {r.duration_s:>9.2f}s  {alert}"
        )

    print("\nPróxima etapa: split externo 70/30 por sujeito (Etapa 5).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 7. Ponto de entrada da Etapa 4
# ---------------------------------------------------------------------------
def run_stage04_quality(
    config: ExperimentConfig | None = None,
    dataset: LoadedDataset | None = None,
    example_plot_ids: list[str] | None = None,
    max_example_plots: int = 3,
) -> tuple[dict[str, SubjectQualityReport], QualitySummary, list[Path]]:
    """
    Carrega dados (se necessário), analisa qualidade e gera plots de exemplo.

    Não faz split. Não aplica filtros. Inspeciona todos os sujeitos igualmente.
    """
    if config is None:
        config = build_default_config()

    paths = create_output_dirs(config.output_dir)
    plots_dir = paths["plots"]

    if dataset is None:
        dataset = load_all_subjects(config.data_dir)

    reports, summary = analyze_all_subjects(
        dataset,
        expected_fs_hz=config.sampling_hz,
    )

    saved_plots = plot_example_subjects(
        dataset,
        output_dir=plots_dir,
        subject_ids=example_plot_ids,
        max_plots=max_example_plots,
    )

    print_quality_report(reports, summary)
    if saved_plots:
        print("\nPlots salvos:")
        for p in saved_plots:
            print(f"  {p}")

    return reports, summary, saved_plots


if __name__ == "__main__":
    run_stage04_quality()
