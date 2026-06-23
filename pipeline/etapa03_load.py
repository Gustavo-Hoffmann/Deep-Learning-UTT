#!/usr/bin/env python3
"""
Etapa 3 — Leitura dos arquivos por sujeito
==========================================
Lê todos os arquivos da pasta de dados, identifica cada sujeito,
valida colunas, resume tamanhos e valores ausentes.

NÃO faz split. NÃO normaliza. NÃO cria janelas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config
from etapa02_schema import (
    REQUIRED_COLUMNS,
    SubjectFileSpec,
    extract_subject_id,
    list_expected_subject_files,
    rename_columns_to_canonical,
    validate_required_columns_strict,
)

# ---------------------------------------------------------------------------
# 1. Leitura de um arquivo (CSV / XLSX)
# ---------------------------------------------------------------------------
def read_subject_table(filepath: Path | str) -> pd.DataFrame:
    """
    Lê um arquivo de sujeito conforme a extensão.

    Suporta: .csv, .xlsx, .xls
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Extensão não suportada: {suffix} ({path.name})")


# ---------------------------------------------------------------------------
# 2. Estruturas de dados por sujeito
# ---------------------------------------------------------------------------
METADATA_COLUMNS: tuple[str, ...] = ("subject_id", "source_file")


@dataclass
class SubjectRecord:
    """Dados de um único sujeito após leitura e padronização."""

    subject_id: str
    source_file: str
    filepath: Path
    dataframe: pd.DataFrame
    n_rows: int
    duration_s: float | None
    missing_total: int
    missing_by_column: dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SubjectRecord(id={self.subject_id!r}, rows={self.n_rows}, "
            f"missing={self.missing_total}, file={self.source_file!r})"
        )


@dataclass
class LoadSummary:
    """Resumo agregado após carregar todos os sujeitos."""

    n_files_found: int
    n_subjects_loaded: int
    n_rows_total: int
    n_rows_min: int
    n_rows_max: int
    n_rows_median: float
    subjects_with_missing: list[str]
    duplicate_subject_ids: list[str]
    subject_ids: list[str]

    def __repr__(self) -> str:
        return (
            f"LoadSummary(subjects={self.n_subjects_loaded}, "
            f"rows_total={self.n_rows_total}, "
            f"rows_per_subject={self.n_rows_min}..{self.n_rows_max})"
        )


@dataclass
class LoadedDataset:
    """
    Container principal: um registro por sujeito.

    Acesso:
        dataset.subjects["02"].dataframe
        dataset.get_subject_ids()
    """

    subjects: dict[str, SubjectRecord]
    summary: LoadSummary
    data_dir: Path

    def get_subject_ids(self) -> list[str]:
        return sorted(self.subjects.keys())

    def get_dataframe(self, subject_id: str) -> pd.DataFrame:
        return self.subjects[subject_id].dataframe

    def to_combined_dataframe(self) -> pd.DataFrame:
        """Concatena todos os sujeitos (útil para inspeção; não usar para split por janela)."""
        frames = [rec.dataframe for rec in self.subjects.values()]
        return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Carregar um único sujeito
# ---------------------------------------------------------------------------
def load_single_subject(
    filepath: Path | str,
    subject_id_strategy: Literal["numeric_prefix", "stem"] = "numeric_prefix",
) -> SubjectRecord:
    """
    Lê um arquivo, valida colunas, renomeia para padrão canônico e
    associa metadados de sujeito.
    """
    path = Path(filepath)
    subject_id = extract_subject_id(path, strategy=subject_id_strategy)

    raw = read_subject_table(path)
    validate_required_columns_strict(raw, filepath=path)

    df = rename_columns_to_canonical(raw)

    # mantém apenas colunas do contrato + metadados
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["subject_id"] = subject_id
    df["source_file"] = path.name

    # ordena por tempo dentro do sujeito
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    # estatísticas de qualidade
    missing_by_column = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}
    missing_total = int(df[list(REQUIRED_COLUMNS)].isna().sum().sum())

    duration_s: float | None = None
    if len(df) >= 2 and df["time"].notna().all():
        duration_s = float(df["time"].iloc[-1] - df["time"].iloc[0])

    return SubjectRecord(
        subject_id=subject_id,
        source_file=path.name,
        filepath=path,
        dataframe=df,
        n_rows=len(df),
        duration_s=duration_s,
        missing_total=missing_total,
        missing_by_column=missing_by_column,
    )


# ---------------------------------------------------------------------------
# 4. Carregar todos os sujeitos da pasta
# ---------------------------------------------------------------------------
def load_all_subjects(
    data_dir: Path | str,
    subject_id_strategy: Literal["numeric_prefix", "stem"] = "numeric_prefix",
    strict_duplicates: bool = True,
) -> LoadedDataset:
    """
    Lê todos os arquivos suportados na pasta e monta um registro por sujeito.

    Parameters
    ----------
    strict_duplicates :
        Se True, levanta erro quando dois arquivos geram o mesmo subject_id.
    """
    folder = Path(data_dir)
    specs = list_expected_subject_files(folder)

    if not specs:
        raise FileNotFoundError(f"Nenhum arquivo CSV/XLSX encontrado em: {folder}")

    subjects: dict[str, SubjectRecord] = {}
    duplicate_ids: list[str] = []

    for spec in specs:
        record = load_single_subject(spec.filepath, subject_id_strategy=subject_id_strategy)

        if record.subject_id in subjects:
            duplicate_ids.append(record.subject_id)
            if strict_duplicates:
                prev = subjects[record.subject_id].source_file
                raise ValueError(
                    f"subject_id duplicado '{record.subject_id}': "
                    f"{prev} e {record.source_file}"
                )
        subjects[record.subject_id] = record

    row_counts = [rec.n_rows for rec in subjects.values()]
    subjects_with_missing = [
        sid for sid, rec in subjects.items() if rec.missing_total > 0
    ]

    summary = LoadSummary(
        n_files_found=len(specs),
        n_subjects_loaded=len(subjects),
        n_rows_total=sum(row_counts),
        n_rows_min=min(row_counts),
        n_rows_max=max(row_counts),
        n_rows_median=float(pd.Series(row_counts).median()),
        subjects_with_missing=subjects_with_missing,
        duplicate_subject_ids=sorted(set(duplicate_ids)),
        subject_ids=sorted(subjects.keys()),
    )

    return LoadedDataset(subjects=subjects, summary=summary, data_dir=folder)


# ---------------------------------------------------------------------------
# 5. Relatório legível
# ---------------------------------------------------------------------------
def print_load_report(dataset: LoadedDataset) -> None:
    """Imprime resumo da carga sem alterar os dados."""
    s = dataset.summary

    print("=" * 60)
    print("ETAPA 3 — Leitura dos arquivos por sujeito")
    print("=" * 60)
    print(f"Pasta de dados : {dataset.data_dir}")
    print(f"Arquivos lidos : {s.n_files_found}")
    print(f"Sujeitos       : {s.n_subjects_loaded}")
    print(f"IDs            : {', '.join(s.subject_ids)}")
    print(f"Amostras total : {s.n_rows_total}")
    print(f"Amostras/sujeito: min={s.n_rows_min}, mediana={s.n_rows_median:.0f}, max={s.n_rows_max}")

    if s.duplicate_subject_ids:
        print(f"\n⚠ subject_id duplicados: {s.duplicate_subject_ids}")

    if s.subjects_with_missing:
        print(f"\n⚠ Sujeitos com valores ausentes ({len(s.subjects_with_missing)}):")
        for sid in s.subjects_with_missing:
            rec = dataset.subjects[sid]
            cols = {k: v for k, v in rec.missing_by_column.items() if v > 0}
            print(f"    {sid}: {cols}")
    else:
        print("\n✓ Nenhum valor ausente nas colunas obrigatórias.")

    print("\nDetalhe por sujeito:")
    print(f"  {'ID':<6} {'arquivo':<22} {'linhas':>8} {'duração(s)':>12} {'missing':>8}")
    print("  " + "-" * 62)
    for sid in s.subject_ids:
        rec = dataset.subjects[sid]
        dur = f"{rec.duration_s:.2f}" if rec.duration_s is not None else "n/a"
        print(
            f"  {sid:<6} {rec.source_file:<22} {rec.n_rows:>8} {dur:>12} {rec.missing_total:>8}"
        )

    print("\nColunas em cada DataFrame (canônicas + metadados):")
    example = dataset.subjects[s.subject_ids[0]].dataframe
    print(f"  {list(example.columns)}")
    print("\nPróxima etapa: conferência de qualidade dos sinais (Etapa 4).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Ponto de entrada da Etapa 3
# ---------------------------------------------------------------------------
def run_stage03_load(config: ExperimentConfig | None = None) -> LoadedDataset:
    """Carrega todos os sujeitos e imprime relatório de qualidade inicial."""
    if config is None:
        config = build_default_config()

    dataset = load_all_subjects(config.data_dir)
    print_load_report(dataset)
    return dataset


if __name__ == "__main__":
    run_stage03_load()
