#!/usr/bin/env python3
"""
Etapa 6 — Normalização sem vazamento
====================================
Ajusta escaladores APENAS nos sujeitos de treino de cada fold e aplica
os mesmos parâmetros em validação/teste.

NÃO cria janelas. NÃO treina modelo. NÃO ajusta escalador com sujeitos de teste final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa02_schema import INPUT_FEATURE_COLUMNS, TARGET_COLUMN
from etapa03_load import LoadedDataset, LoadSummary, SubjectRecord, load_all_subjects
from etapa05_split import (
    SubjectSplit,
    apply_split_to_dataset,
    load_subject_split,
)

METADATA_COLUMNS: tuple[str, ...] = ("subject_id", "source_file")


# ---------------------------------------------------------------------------
# 1. Pacote de escaladores
# ---------------------------------------------------------------------------
@dataclass
class ScalerBundle:
    """
    Escaladores ajustados em um conjunto de treino específico.

    feature_scaler : StandardScaler nas 6 entradas IMU
    target_scaler  : opcional, para a coluna vicon (alvo)
    """

    feature_scaler: StandardScaler
    feature_cols: list[str]
    target_col: str = TARGET_COLUMN
    target_scaler: StandardScaler | None = None
    fitted_on_subject_ids: list[str] | None = None

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(values)

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            raise ValueError("target_scaler não foi ajustado (normalize_target=False).")
        return self.target_scaler.transform(values.reshape(-1, 1)).ravel()

    def inverse_transform_target(self, values_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte escala do alvo para unidades originais (ex.: cm).

        Use após predição para calcular MAE/RMSE interpretáveis.
        """
        if self.target_scaler is None:
            return np.asarray(values_scaled, dtype=float)
        arr = np.asarray(values_scaled, dtype=float).reshape(-1, 1)
        return self.target_scaler.inverse_transform(arr).ravel()

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str) -> ScalerBundle:
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 2. Coleta de matrizes para ajuste
# ---------------------------------------------------------------------------
def collect_columns_matrix(
    dataset: LoadedDataset,
    subject_ids: list[str],
    columns: list[str],
) -> np.ndarray:
    """Empilha valores das colunas indicadas a partir dos sujeitos de treino."""
    chunks: list[np.ndarray] = []
    for sid in subject_ids:
        df = dataset.subjects[sid].dataframe
        chunks.append(df[columns].to_numpy(dtype=float))
    if not chunks:
        raise ValueError("Nenhum sujeito fornecido para ajuste do escalador.")
    return np.vstack(chunks)


def channel_stats(matrix: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Resumo por canal (média e desvio) — útil para inspeção didática."""
    return pd.DataFrame(
        {
            "canal": names,
            "media": matrix.mean(axis=0),
            "desvio": matrix.std(axis=0),
        }
    )


# ---------------------------------------------------------------------------
# 3. Ajuste e aplicação sem vazamento
# ---------------------------------------------------------------------------
def fit_scaler_bundle(
    dataset: LoadedDataset,
    train_subject_ids: list[str],
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
    normalize_target: bool = True,
) -> ScalerBundle:
    """
    Ajusta escaladores usando APENAS sujeitos de treino do fold.

    Parameters
    ----------
    train_subject_ids :
        IDs dos sujeitos cujas amostras entram no `.fit()`.
    normalize_target :
        Se True, também ajusta StandardScaler no vicon.
    """
    features = list(feature_cols or INPUT_FEATURE_COLUMNS)
    x_train = collect_columns_matrix(dataset, train_subject_ids, features)

    feature_scaler = StandardScaler()
    feature_scaler.fit(x_train)

    target_scaler = None
    if normalize_target:
        y_train = collect_columns_matrix(dataset, train_subject_ids, [target_col])
        target_scaler = StandardScaler()
        target_scaler.fit(y_train)

    return ScalerBundle(
        feature_scaler=feature_scaler,
        feature_cols=features,
        target_col=target_col,
        target_scaler=target_scaler,
        fitted_on_subject_ids=sorted(train_subject_ids),
    )


def transform_subject_record(
    record: SubjectRecord,
    bundle: ScalerBundle,
    normalize_target: bool = True,
) -> SubjectRecord:
    """Aplica escaladores já ajustados a um único sujeito."""
    df = record.dataframe.copy()

    x = df[bundle.feature_cols].to_numpy(dtype=float)
    df[bundle.feature_cols] = bundle.transform_features(x)

    if normalize_target and bundle.target_scaler is not None:
        y = df[bundle.target_col].to_numpy(dtype=float)
        df[bundle.target_col] = bundle.transform_target(y)

    return SubjectRecord(
        subject_id=record.subject_id,
        source_file=record.source_file,
        filepath=record.filepath,
        dataframe=df,
        n_rows=record.n_rows,
        duration_s=record.duration_s,
        missing_total=record.missing_total,
        missing_by_column=record.missing_by_column,
    )


def transform_dataset(
    dataset: LoadedDataset,
    bundle: ScalerBundle,
    subject_ids: list[str] | None = None,
    normalize_target: bool = True,
) -> LoadedDataset:
    """Aplica transformação a um subconjunto de sujeitos (sem reajustar)."""
    ids = subject_ids or dataset.get_subject_ids()
    subjects = {
        sid: transform_subject_record(dataset.subjects[sid], bundle, normalize_target=normalize_target)
        for sid in ids
    }
    row_counts = [rec.n_rows for rec in subjects.values()]

    summary = LoadSummary(
        n_files_found=len(subjects),
        n_subjects_loaded=len(subjects),
        n_rows_total=sum(row_counts),
        n_rows_min=min(row_counts) if row_counts else 0,
        n_rows_max=max(row_counts) if row_counts else 0,
        n_rows_median=float(pd.Series(row_counts).median()) if row_counts else 0.0,
        subjects_with_missing=[sid for sid, rec in subjects.items() if rec.missing_total > 0],
        duplicate_subject_ids=[],
        subject_ids=sorted(ids),
    )
    return LoadedDataset(subjects=subjects, summary=summary, data_dir=dataset.data_dir)


# ---------------------------------------------------------------------------
# 4. Normalização manual por canal (alternativa ao StandardScaler)
# ---------------------------------------------------------------------------
@dataclass
class ManualChannelScaler:
    """z-score manual: (x - mean) / std, parâmetros fixos por canal."""

    means: np.ndarray
    stds: np.ndarray
    feature_cols: list[str]

    @classmethod
    def fit(cls, matrix: np.ndarray, feature_cols: list[str], eps: float = 1e-8) -> ManualChannelScaler:
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        stds = np.where(stds < eps, 1.0, stds)
        return cls(means=means, stds=stds, feature_cols=feature_cols)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.means) / self.stds


def fit_manual_feature_scaler(
    dataset: LoadedDataset,
    train_subject_ids: list[str],
    feature_cols: list[str] | None = None,
) -> ManualChannelScaler:
    """Equivalente manual ao StandardScaler de features."""
    features = list(feature_cols or INPUT_FEATURE_COLUMNS)
    x_train = collect_columns_matrix(dataset, train_subject_ids, features)
    return ManualChannelScaler.fit(x_train, features)


# ---------------------------------------------------------------------------
# 5. Um fold (ex.: LOSO dentro dos 70% de desenvolvimento)
# ---------------------------------------------------------------------------
@dataclass
class FoldNormalizationResult:
    fold_name: str
    train_subject_ids: list[str]
    val_subject_ids: list[str]
    bundle: ScalerBundle
    train_normalized: LoadedDataset
    val_normalized: LoadedDataset


def normalize_loso_fold(
    dev_dataset: LoadedDataset,
    val_subject_id: str,
    normalize_target: bool = True,
) -> FoldNormalizationResult:
    """
    Exemplo de normalização para um fold LOSO no grupo de desenvolvimento.

    - Ajusta escalador nos sujeitos de treino do fold.
    - Aplica no sujeito de validação sem incluí-lo no `.fit()`.
    """
    all_dev_ids = dev_dataset.get_subject_ids()
    if val_subject_id not in all_dev_ids:
        raise KeyError(f"Sujeito {val_subject_id!r} não está no grupo de desenvolvimento.")

    train_ids = sorted([sid for sid in all_dev_ids if sid != val_subject_id])
    val_ids = [val_subject_id]

    bundle = fit_scaler_bundle(
        dev_dataset,
        train_subject_ids=train_ids,
        normalize_target=normalize_target,
    )

    train_norm = transform_dataset(dev_dataset, bundle, subject_ids=train_ids, normalize_target=normalize_target)
    val_norm = transform_dataset(dev_dataset, bundle, subject_ids=val_ids, normalize_target=normalize_target)

    return FoldNormalizationResult(
        fold_name=f"loso_val_{val_subject_id}",
        train_subject_ids=train_ids,
        val_subject_ids=val_ids,
        bundle=bundle,
        train_normalized=train_norm,
        val_normalized=val_norm,
    )


# ---------------------------------------------------------------------------
# 6. Demonstração do erro de vazamento (didático)
# ---------------------------------------------------------------------------
def demonstrate_leakage_contrast(
    dev_dataset: LoadedDataset,
    val_subject_id: str,
    feature_cols: list[str] | None = None,
) -> None:
    """
    Mostra por que ajustar o scaler com TODOS os sujeitos (incluindo validação)
    é incorreto.
    """
    features = list(feature_cols or INPUT_FEATURE_COLUMNS)
    all_ids = dev_dataset.get_subject_ids()
    train_ids = sorted([sid for sid in all_ids if sid != val_subject_id])

    # ERRADO: inclui sujeito de validação no ajuste
    wrong = fit_scaler_bundle(dev_dataset, train_subject_ids=all_ids, normalize_target=False)

    # CERTO: apenas treino do fold
    right = fit_scaler_bundle(dev_dataset, train_subject_ids=train_ids, normalize_target=False)

    val_matrix = dev_dataset.subjects[val_subject_id].dataframe[features].to_numpy(dtype=float)

    wrong_scaled = wrong.transform_features(val_matrix)
    right_scaled = right.transform_features(val_matrix)

    print("\nContraste de vazamento (features do sujeito de validação após escalar):")
    print(f"  Sujeito de validação: {val_subject_id}")
    print(
        f"  Média (ERRADO — fit com validação): {wrong_scaled.mean(axis=0).round(4)}"
    )
    print(
        f"  Média (CERTO  — fit só treino)    : {right_scaled.mean(axis=0).round(4)}"
    )
    print(
        "  No procedimento correto, a validação NÃO precisa ter média ~0 "
        "canal a canal (isso é esperado)."
    )


# ---------------------------------------------------------------------------
# 7. Relatório legível
# ---------------------------------------------------------------------------
def print_normalization_report(
    fold: FoldNormalizationResult,
    normalize_target: bool,
) -> None:
    features = fold.bundle.feature_cols
    sid_val = fold.val_subject_ids[0]

    raw_val = fold.val_normalized.subjects[sid_val].dataframe  # já transformado
    # recupera estatísticas comparando com dev original via bundle
    print("=" * 60)
    print("ETAPA 6 — Normalização sem vazamento")
    print("=" * 60)
    print(f"Fold exemplo     : {fold.fold_name}")
    print(f"Treino (fit)     : {len(fold.train_subject_ids)} sujeitos -> {fold.train_subject_ids[:5]}...")
    print(f"Validação        : {fold.val_subject_ids}")
    print(f"Features         : {features}")
    print(f"Alvo normalizado : {normalize_target}")

    train_mat = collect_columns_matrix(fold.train_normalized, fold.train_subject_ids, features)
    val_mat = collect_columns_matrix(fold.val_normalized, fold.val_subject_ids, features)

    print("\nEstatísticas APÓS normalizar (fold correto):")
    print("  Treino (deve ter média ~0, desvio ~1 por canal):")
    print(channel_stats(train_mat, features).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    print("\n  Validação (parâmetros do treino; média pode ≠ 0):")
    print(channel_stats(val_mat, features).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    if normalize_target and fold.bundle.target_scaler is not None:
        y_scaled = fold.val_normalized.subjects[sid_val].dataframe[TARGET_COLUMN].to_numpy()
        y_pred_scaled = y_scaled[:3]  # exemplo ilustrativo: "predições" fictícias
        y_true_cm = fold.bundle.inverse_transform_target(y_scaled[:3])
        print("\nInversão do alvo (exemplo ilustrativo com 3 pontos do val):")
        print(f"  vicon escalado : {y_pred_scaled.round(4)}")
        print(f"  vicon em cm    : {y_true_cm.round(4)}  (unidade original)")

    print("\nRegras:")
    print("  - NUNCA incluir sujeito de validação/teste no `.fit()`.")
    print("  - Sujeitos de teste final (30%): escalador ajustado só na Etapa 15.")
    print("\nPróxima etapa: criação de janelas temporais (Etapa 7).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 8. Ponto de entrada da Etapa 6
# ---------------------------------------------------------------------------
def run_stage06_normalize(
    config: ExperimentConfig | None = None,
    dataset: LoadedDataset | None = None,
    split: SubjectSplit | None = None,
    example_val_subject_id: str = "02",
    normalize_target: bool = True,
) -> FoldNormalizationResult:
    """
    Demonstra normalização correta em um fold LOSO de desenvolvimento.

    Usa apenas sujeitos de desenvolvimento. Não toca no teste final.
    """
    if config is None:
        config = build_default_config()

    paths = create_output_dirs(config.output_dir)

    if dataset is None:
        dataset = load_all_subjects(config.data_dir)

    if split is None:
        split = load_subject_split(paths["splits"])

    dev_dataset, _test_dataset = apply_split_to_dataset(dataset, split)

    if example_val_subject_id not in dev_dataset.subjects:
        example_val_subject_id = dev_dataset.get_subject_ids()[0]

    fold = normalize_loso_fold(
        dev_dataset,
        val_subject_id=example_val_subject_id,
        normalize_target=normalize_target,
    )

    demonstrate_leakage_contrast(dev_dataset, val_subject_id=example_val_subject_id)
    print_normalization_report(fold, normalize_target=normalize_target)

    scaler_path = fold.bundle.save(paths["scalers"] / f"{fold.fold_name}_scaler.joblib")
    print(f"\nScaler do fold exemplo salvo em: {scaler_path}")

    meta = {
        "fold_name": fold.fold_name,
        "train_subject_ids": fold.train_subject_ids,
        "val_subject_ids": fold.val_subject_ids,
        "feature_cols": fold.bundle.feature_cols,
        "normalize_target": normalize_target,
        "scaler_path": str(scaler_path),
        "note": "Teste final (30%) não foi usado no fit.",
    }
    meta_path = paths["scalers"] / f"{fold.fold_name}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return fold


if __name__ == "__main__":
    run_stage06_normalize()
