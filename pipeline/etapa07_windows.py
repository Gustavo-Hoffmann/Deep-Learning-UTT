#!/usr/bin/env python3
"""
Etapa 7 — Criação de janelas temporais
======================================
Transforma o sinal contínuo de cada sujeito em janelas para PyTorch.

Formato de entrada X : (n_janelas, n_canais, tamanho_janela)
Alvo y (amplitude)   : (n_janelas,)  — max(vicon) - min(vicon) por janela
Alvo y (curva)       : (n_janelas, tamanho_janela)

NÃO treina modelo. NÃO refaz split. Mantém subject_id por janela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa02_schema import INPUT_FEATURE_COLUMNS, TARGET_COLUMN, TIME_COLUMN, TargetMode
from etapa03_load import LoadedDataset, SubjectRecord, load_all_subjects
from etapa05_split import SubjectSplit, apply_split_to_dataset, load_subject_split
from etapa06_normalize import normalize_loso_fold

# ---------------------------------------------------------------------------
# 1. Estrutura de janelas
# ---------------------------------------------------------------------------
@dataclass
class WindowBatch:
    """
    Lote de janelas extraídas de um ou mais sujeitos.

    X : (n_janelas, n_canais, tamanho_janela) — pronto para Conv1d / TCN
    y : amplitude -> (n_janelas,)
        curva     -> (n_janelas, tamanho_janela)
    """

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    window_start_idx: np.ndarray
    window_start_time: np.ndarray
    target_mode: TargetMode
    feature_cols: list[str]
    window_samples: int
    stride_samples: int

    @property
    def n_windows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])

    @property
    def window_size(self) -> int:
        return int(self.X.shape[2])

    def summary(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "X_shape": tuple(self.X.shape),
            "y_shape": tuple(self.y.shape),
            "target_mode": self.target_mode,
            "subjects": sorted(set(self.subject_ids.tolist())),
        }


# ---------------------------------------------------------------------------
# 2. Cálculo do alvo por janela
# ---------------------------------------------------------------------------
def compute_window_target(
    vicon_window: np.ndarray,
    target_mode: TargetMode = "amplitude",
) -> float | np.ndarray:
    """
    Calcula o alvo a partir do Vicon dentro da janela.

    amplitude : max(vicon) - min(vicon)  — escalar por janela
    curve     : vetor temporal completo do Vicon na janela
    """
    v = np.asarray(vicon_window, dtype=float)
    if target_mode == "amplitude":
        return float(np.max(v) - np.min(v))
    if target_mode == "curve":
        return v.copy()
    raise ValueError(f"target_mode desconhecido: {target_mode!r}")


# ---------------------------------------------------------------------------
# 3. Extração de janelas de um sujeito
# ---------------------------------------------------------------------------
def iter_window_starts(n_samples: int, window_samples: int, stride_samples: int) -> list[int]:
    """
    Gera índices iniciais de janelas deslizantes.

    stride < window_samples  -> janelas sobrepostas
    stride == window_samples -> janelas contíguas, sem sobreposição
    """
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_samples e stride_samples devem ser positivos.")
    if n_samples < window_samples:
        return []

    starts: list[int] = []
    start = 0
    while start + window_samples <= n_samples:
        starts.append(start)
        start += stride_samples
    return starts


def create_windows_from_dataframe(
    df: pd.DataFrame,
    subject_id: str,
    window_samples: int,
    stride_samples: int,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
    time_col: str = TIME_COLUMN,
    target_mode: TargetMode = "amplitude",
) -> WindowBatch:
    """
    Extrai janelas deslizantes de um DataFrame de um único sujeito.

    Cada linha do DataFrame = um instante de amostragem alinhado.
    """
    features = list(feature_cols or INPUT_FEATURE_COLUMNS)
    n_samples = len(df)
    starts = iter_window_starts(n_samples, window_samples, stride_samples)

    if not starts:
        empty_x = np.empty((0, len(features), window_samples), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32) if target_mode == "amplitude" else np.empty((0, window_samples), dtype=np.float32)
        return WindowBatch(
            X=empty_x,
            y=empty_y,
            subject_ids=np.empty((0,), dtype=object),
            window_start_idx=np.empty((0,), dtype=int),
            window_start_time=np.empty((0,), dtype=float),
            target_mode=target_mode,
            feature_cols=features,
            window_samples=window_samples,
            stride_samples=stride_samples,
        )

    x_mat = df[features].to_numpy(dtype=np.float32)
    vicon = df[target_col].to_numpy(dtype=np.float32)
    times = df[time_col].to_numpy(dtype=float)

    n_windows = len(starts)
    n_channels = len(features)
    X = np.empty((n_windows, n_channels, window_samples), dtype=np.float32)

    if target_mode == "amplitude":
        y = np.empty((n_windows,), dtype=np.float32)
    else:
        y = np.empty((n_windows, window_samples), dtype=np.float32)

    subject_ids = np.empty((n_windows,), dtype=object)
    start_idx = np.empty((n_windows,), dtype=int)
    start_time = np.empty((n_windows,), dtype=float)

    for i, s in enumerate(starts):
        e = s + window_samples
        X[i] = x_mat[s:e].T  # (canais, tempo) — layout PyTorch Conv1d
        target = compute_window_target(vicon[s:e], target_mode=target_mode)
        y[i] = target
        subject_ids[i] = subject_id
        start_idx[i] = s
        start_time[i] = times[s]

    return WindowBatch(
        X=X,
        y=y,
        subject_ids=subject_ids,
        window_start_idx=start_idx,
        window_start_time=start_time,
        target_mode=target_mode,
        feature_cols=features,
        window_samples=window_samples,
        stride_samples=stride_samples,
    )


def create_windows_from_subject(
    record: SubjectRecord,
    window_samples: int,
    stride_samples: int,
    feature_cols: list[str] | None = None,
    target_mode: TargetMode = "amplitude",
) -> WindowBatch:
    """Atalho para extrair janelas de um ``SubjectRecord``."""
    return create_windows_from_dataframe(
        df=record.dataframe,
        subject_id=record.subject_id,
        window_samples=window_samples,
        stride_samples=stride_samples,
        feature_cols=feature_cols,
        target_mode=target_mode,
    )


def create_windows_from_dataset(
    dataset: LoadedDataset,
    window_samples: int,
    stride_samples: int,
    feature_cols: list[str] | None = None,
    target_mode: TargetMode = "amplitude",
    subject_ids: list[str] | None = None,
) -> WindowBatch:
    """
    Extrai janelas de vários sujeitos e concatena.

    Cada janela mantém o ``subject_id`` de origem — essencial para
    não misturar sujeitos entre treino/validação/teste.
    """
    ids = subject_ids or dataset.get_subject_ids()
    batches: list[WindowBatch] = []

    for sid in ids:
        batch = create_windows_from_subject(
            dataset.subjects[sid],
            window_samples=window_samples,
            stride_samples=stride_samples,
            feature_cols=feature_cols,
            target_mode=target_mode,
        )
        if batch.n_windows > 0:
            batches.append(batch)

    if not batches:
        features = list(feature_cols or INPUT_FEATURE_COLUMNS)
        empty_x = np.empty((0, len(features), window_samples), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32) if target_mode == "amplitude" else np.empty((0, window_samples), dtype=np.float32)
        return WindowBatch(
            X=empty_x,
            y=empty_y,
            subject_ids=np.empty((0,), dtype=object),
            window_start_idx=np.empty((0,), dtype=int),
            window_start_time=np.empty((0,), dtype=float),
            target_mode=target_mode,
            feature_cols=features,
            window_samples=window_samples,
            stride_samples=stride_samples,
        )

    return concat_window_batches(batches)


def concat_window_batches(batches: list[WindowBatch]) -> WindowBatch:
    """Concatena lotes de janelas (mesmos hiperparâmetros de janela)."""
    if not batches:
        raise ValueError("Lista de batches vazia.")

    ref = batches[0]
    for b in batches[1:]:
        if (b.window_samples, b.stride_samples, b.target_mode, b.feature_cols) != (
            ref.window_samples,
            ref.stride_samples,
            ref.target_mode,
            ref.feature_cols,
        ):
            raise ValueError("Todos os batches devem compartilhar os mesmos parâmetros de janela.")

    return WindowBatch(
        X=np.concatenate([b.X for b in batches], axis=0),
        y=np.concatenate([b.y for b in batches], axis=0),
        subject_ids=np.concatenate([b.subject_ids for b in batches], axis=0),
        window_start_idx=np.concatenate([b.window_start_idx for b in batches], axis=0),
        window_start_time=np.concatenate([b.window_start_time for b in batches], axis=0),
        target_mode=ref.target_mode,
        feature_cols=ref.feature_cols,
        window_samples=ref.window_samples,
        stride_samples=ref.stride_samples,
    )


# ---------------------------------------------------------------------------
# 4. Utilitários de integridade
# ---------------------------------------------------------------------------
def count_windows_per_subject(batch: WindowBatch) -> dict[str, int]:
    """Conta janelas por sujeito."""
    counts: dict[str, int] = {}
    for sid in batch.subject_ids:
        counts[sid] = counts.get(sid, 0) + 1
    return dict(sorted(counts.items()))


def assert_subject_isolation(
    train_batch: WindowBatch,
    val_batch: WindowBatch,
) -> None:
    """Garante que nenhum subject_id aparece em treino e validação ao mesmo tempo."""
    train_subjects = set(train_batch.subject_ids.tolist())
    val_subjects = set(val_batch.subject_ids.tolist())
    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(
            "Vazamento: sujeitos presentes em treino E validação: "
            f"{sorted(overlap)}"
        )


def windows_per_subject_formula(n_samples: int, window_samples: int, stride_samples: int) -> int:
    """Número teórico de janelas para um sujeito."""
    return len(iter_window_starts(n_samples, window_samples, stride_samples))


# ---------------------------------------------------------------------------
# 5. Relatório legível
# ---------------------------------------------------------------------------
def print_window_report(
    config: ExperimentConfig,
    train_batch: WindowBatch,
    val_batch: WindowBatch,
    val_subject_id: str,
) -> None:
    overlap_pct = (1.0 - config.stride_seconds / config.window_seconds) * 100

    print("=" * 60)
    print("ETAPA 7 — Criação de janelas temporais")
    print("=" * 60)
    print(f"Frequência        : {config.sampling_hz} Hz")
    print(f"Janela            : {config.window_seconds} s = {config.window_samples} amostras")
    print(f"Stride            : {config.stride_seconds} s = {config.stride_samples} amostras")
    print(f"Sobreposição      : {overlap_pct:.0f}%")
    print(f"Canais de entrada : {config.feature_cols}")
    print(f"Modo de alvo      : {config.target_mode}")

    print("\nFormato PyTorch (treino do fold exemplo):")
    print(f"  X : {train_batch.X.shape}  -> (n_janelas, canais, tempo)")
    print(f"  y : {train_batch.y.shape}")

    print("\nFormato PyTorch (validação do fold exemplo):")
    print(f"  X : {val_batch.X.shape}")
    print(f"  y : {val_batch.y.shape}")

    print(f"\nJanelas por sujeito (treino) — primeiros 5:")
    train_counts = count_windows_per_subject(train_batch)
    for sid, n in list(train_counts.items())[:5]:
        print(f"  {sid}: {n} janelas")
    if len(train_counts) > 5:
        print(f"  ... (+{len(train_counts) - 5} sujeitos)")

    n_val = count_windows_per_subject(val_batch).get(val_subject_id, 0)
    print(f"\nValidação (sujeito {val_subject_id}): {n_val} janelas")

    if config.target_mode == "amplitude":
        print("\nAlvo amplitude (3 primeiras janelas de validação):")
        for i in range(min(3, val_batch.n_windows)):
            print(
                f"  janela {i}: subject={val_batch.subject_ids[i]}, "
                f"t0={val_batch.window_start_time[i]:.3f}s, "
                f"y={val_batch.y[i]:.4f}"
            )

    print("\nOrdem correta no pipeline:")
    print("  1. split por sujeito (Etapa 5)")
    print("  2. normalizar treino do fold + transformar val (Etapa 6)")
    print("  3. criar janelas separadamente em treino e val (esta etapa)")
    print("\nPróxima etapa: Dataset e DataLoader PyTorch (Etapa 8).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Ponto de entrada da Etapa 7
# ---------------------------------------------------------------------------
def run_stage07_windows(
    config: ExperimentConfig | None = None,
    example_val_subject_id: str = "02",
    target_mode: TargetMode | None = None,
) -> tuple[WindowBatch, WindowBatch]:
    """
    Demonstra janelamento em um fold LOSO de desenvolvimento.

    Fluxo: carrega dados -> split dev/test -> normaliza fold -> janelas treino/val.
    Sujeitos de teste final (30%) não são usados.
    """
    if config is None:
        config = build_default_config()

    paths = create_output_dirs(config.output_dir)
    mode: TargetMode = target_mode or config.target_mode

    dataset = load_all_subjects(config.data_dir)
    split = load_subject_split(paths["splits"])
    dev_dataset, _ = apply_split_to_dataset(dataset, split)

    if example_val_subject_id not in dev_dataset.subjects:
        example_val_subject_id = dev_dataset.get_subject_ids()[0]

    fold = normalize_loso_fold(dev_dataset, val_subject_id=example_val_subject_id)

    train_batch = create_windows_from_dataset(
        fold.train_normalized,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=mode,
    )
    val_batch = create_windows_from_dataset(
        fold.val_normalized,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=mode,
    )

    assert_subject_isolation(train_batch, val_batch)

    print_window_report(config, train_batch, val_batch, example_val_subject_id)
    return train_batch, val_batch


if __name__ == "__main__":
    run_stage07_windows()
