#!/usr/bin/env python3
"""
Etapa 2 — Estrutura esperada dos arquivos
=========================================
Define o contrato de dados (um arquivo = um sujeito), extrai subject_id
do nome do arquivo, diferencia modos de alvo e valida colunas obrigatórias.

NÃO carrega todos os arquivos. NÃO faz split. NÃO normaliza.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config

# ---------------------------------------------------------------------------
# 1. Contrato canônico de colunas
# ---------------------------------------------------------------------------
# Nomes padronizados usados internamente pelo pipeline.
# Arquivos reais podem ter nomes diferentes; veja COLUMN_ALIASES abaixo.

REQUIRED_COLUMNS: tuple[str, ...] = (
    "time",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "vicon",
)

INPUT_FEATURE_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS[1:7]
TARGET_COLUMN: str = "vicon"
TIME_COLUMN: str = "time"

# Mapeamento: nome no arquivo real -> nome canônico do pipeline
# Ajuste aqui se novos arquivos usarem outros rótulos.
COLUMN_ALIASES: dict[str, str] = {
    # tempo
    "Time": "time",
    "time": "time",
    "tempo_norm_s": "time",
    # acelerômetro
    "accX_m_s2": "acc_x",
    "accY_m_s2": "acc_y",
    "accZ_m_s2": "acc_z",
    "acc_x": "acc_x",
    "acc_y": "acc_y",
    "acc_z": "acc_z",
    # giroscópio
    "gyroX_rad_s": "gyro_x",
    "gyroY_rad_s": "gyro_y",
    "gyroZ_rad_s": "gyro_z",
    "gyro_x": "gyro_x",
    "gyro_y": "gyro_y",
    "gyro_z": "gyro_z",
    # referência Vicon
    "vicon_esternoZ_cm": "vicon",
    "vicon_esternoZ_mm_norm": "vicon",
    "vicon": "vicon",
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".csv", ".xlsx", ".xls")


# ---------------------------------------------------------------------------
# 2. Modo de predição do alvo
# ---------------------------------------------------------------------------
TargetMode = Literal["amplitude", "curve"]


@dataclass(frozen=True)
class TargetModeInfo:
    """Descrição dos dois modos de predição dentro de cada janela temporal."""

    mode: TargetMode
    summary: str
    y_shape_example: str
    loss_hint: str


TARGET_MODE_DOCS: dict[TargetMode, TargetModeInfo] = {
    "amplitude": TargetModeInfo(
        mode="amplitude",
        summary=(
            "Um único número por janela: amplitude do deslocamento Vicon "
            "naquele intervalo (ex.: max(vicon) - min(vicon))."
        ),
        y_shape_example="(n_janelas, 1)  — regressão escalar",
        loss_hint="MSE / MAE / Huber entre valor predito e amplitude real",
    ),
    "curve": TargetModeInfo(
        mode="curve",
        summary=(
            "O vetor temporal completo do Vicon dentro da janela "
            "(um valor por instante de amostragem)."
        ),
        y_shape_example="(n_janelas, tamanho_janela)  — regressão sequencial",
        loss_hint="MSE/MAE média sobre todos os instantes da janela",
    ),
}


# ---------------------------------------------------------------------------
# 3. Metadados de um arquivo de sujeito
# ---------------------------------------------------------------------------
@dataclass
class SubjectFileSpec:
    """Descrição de um arquivo esperado na pasta de dados."""

    filepath: Path
    subject_id: str
    extension: str
    expected_columns: tuple[str, ...] = field(default_factory=lambda: REQUIRED_COLUMNS)

    @property
    def filename(self) -> str:
        return self.filepath.name


# ---------------------------------------------------------------------------
# 4. Extração de subject_id a partir do nome do arquivo
# ---------------------------------------------------------------------------
def extract_subject_id(filepath: Path | str, strategy: Literal["numeric_prefix", "stem"] = "numeric_prefix") -> str:
    """
    Deriva o identificador do sujeito a partir do nome do arquivo.

    Estratégias
    -----------
    numeric_prefix :
        Usa o prefixo numérico inicial do stem.
        Ex.: ``02_alinhado_ml.csv`` -> ``"02"``
    stem :
        Usa o nome completo sem extensão.
        Ex.: ``02_alinhado_ml.csv`` -> ``"02_alinhado_ml"``

    Use ``stem`` quando houver risco de colisão entre prefixos numéricos.
    """
    path = Path(filepath)
    stem = path.stem

    if strategy == "stem":
        return stem

    match = re.match(r"^(\d+)", stem)
    if match:
        return match.group(1)

    return stem


def build_subject_file_spec(filepath: Path | str, strategy: Literal["numeric_prefix", "stem"] = "numeric_prefix") -> SubjectFileSpec:
    """Monta metadados de um arquivo de sujeito sem ler o conteúdo."""
    path = Path(filepath)
    return SubjectFileSpec(
        filepath=path,
        subject_id=extract_subject_id(path, strategy=strategy),
        extension=path.suffix.lower(),
    )


def list_expected_subject_files(data_dir: Path | str) -> list[SubjectFileSpec]:
    """
    Lista arquivos suportados na pasta de dados e associa subject_id.

    Não abre os arquivos — apenas inspeciona nomes e extensões.
    """
    folder = Path(data_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"Pasta de dados não encontrada: {folder}")

    specs: list[SubjectFileSpec] = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            specs.append(build_subject_file_spec(path))

    return specs


# ---------------------------------------------------------------------------
# 5. Normalização de nomes de colunas
# ---------------------------------------------------------------------------
def rename_columns_to_canonical(df: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Renomeia colunas do arquivo para o padrão canônico do pipeline.

    Colunas sem alias conhecido são mantidas com o nome original.
    """
    mapping = aliases or COLUMN_ALIASES
    rename_map = {col: mapping[col] for col in df.columns if col in mapping}
    return df.rename(columns=rename_map)


def resolve_column_names(
    available_columns: list[str] | pd.Index,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
    aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Encontra, para cada coluna canônica obrigatória, o nome correspondente
    no arquivo (direto ou via alias).

    Retorna ``{nome_canônico: nome_no_arquivo}``.
    """
    mapping = aliases or COLUMN_ALIASES
    available = list(available_columns)

    # índice: nome canônico -> nome original no arquivo
    canonical_to_original: dict[str, str] = {}

    for col in available:
        canonical = mapping.get(col, col)
        canonical_to_original.setdefault(canonical, col)

    resolved: dict[str, str] = {}
    missing: list[str] = []

    for req in required:
        if req in canonical_to_original:
            resolved[req] = canonical_to_original[req]
        else:
            missing.append(req)

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes após mapeamento: "
            f"{missing}. Colunas disponíveis: {available}"
        )

    return resolved


# ---------------------------------------------------------------------------
# 6. Validação de colunas obrigatórias
# ---------------------------------------------------------------------------
@dataclass
class ColumnValidationResult:
    ok: bool
    filepath: Path | None
    subject_id: str | None
    missing_canonical: list[str]
    resolved_mapping: dict[str, str]
    message: str


def validate_required_columns(
    df: pd.DataFrame,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
    aliases: dict[str, str] | None = None,
    filepath: Path | str | None = None,
    subject_id: str | None = None,
) -> ColumnValidationResult:
    """
    Verifica se um DataFrame possui todas as colunas obrigatórias.

    Aceita nomes canônicos ou aliases definidos em COLUMN_ALIASES.
    """
    path = Path(filepath) if filepath is not None else None
    sid = subject_id or (extract_subject_id(path) if path else None)

    try:
        resolved = resolve_column_names(df.columns, required=required, aliases=aliases)
        return ColumnValidationResult(
            ok=True,
            filepath=path,
            subject_id=sid,
            missing_canonical=[],
            resolved_mapping=resolved,
            message="OK: todas as colunas obrigatórias foram encontradas.",
        )
    except ValueError as exc:
        # identifica exatamente quais canônicas faltam
        mapping = aliases or COLUMN_ALIASES
        available_canonical = {mapping.get(c, c) for c in df.columns}
        missing = [c for c in required if c not in available_canonical]
        return ColumnValidationResult(
            ok=False,
            filepath=path,
            subject_id=sid,
            missing_canonical=missing,
            resolved_mapping={},
            message=str(exc),
        )


def validate_required_columns_strict(
    df: pd.DataFrame,
    required: tuple[str, ...] = REQUIRED_COLUMNS,
    aliases: dict[str, str] | None = None,
    filepath: Path | str | None = None,
) -> ColumnValidationResult:
    """Igual a ``validate_required_columns``, mas levanta erro se faltar coluna."""
    result = validate_required_columns(df, required=required, aliases=aliases, filepath=filepath)
    if not result.ok:
        label = result.filepath.name if result.filepath else "DataFrame"
        raise ValueError(f"[{label}] {result.message}")
    return result


# ---------------------------------------------------------------------------
# 7. Exemplo mínimo ilustrativo (não usa seus dados reais)
# ---------------------------------------------------------------------------
def make_minimal_example_dataframe() -> pd.DataFrame:
    """
    DataFrame fictício com 4 amostras para demonstrar validação.

    Serve apenas para ensino — não representa medições reais.
    """
    return pd.DataFrame(
        {
            "time": [0.0, 0.0167, 0.0333, 0.0500],
            "acc_x": [0.1, 0.2, 0.1, 0.0],
            "acc_y": [-0.2, -0.1, 0.0, 0.1],
            "acc_z": [9.8, 9.7, 9.8, 9.8],
            "gyro_x": [0.01, 0.02, 0.01, 0.00],
            "gyro_y": [0.00, 0.01, 0.02, 0.01],
            "gyro_z": [0.00, 0.00, 0.01, 0.01],
            "vicon": [0.0, 0.5, 1.0, 0.8],
        }
    )


def make_minimal_example_with_aliases() -> pd.DataFrame:
    """Exemplo com nomes reais do projeto (sem carregar arquivos)."""
    return pd.DataFrame(
        {
            "Time": [0.0, 0.0167],
            "accX_m_s2": [0.1, 0.2],
            "accY_m_s2": [-0.2, -0.1],
            "accZ_m_s2": [9.8, 9.7],
            "gyroX_rad_s": [0.01, 0.02],
            "gyroY_rad_s": [0.00, 0.01],
            "gyroZ_rad_s": [0.00, 0.00],
            "vicon_esternoZ_cm": [0.0, 0.5],
        }
    )


# ---------------------------------------------------------------------------
# 8. Ponto de entrada da Etapa 2
# ---------------------------------------------------------------------------
def run_stage02_schema(config: ExperimentConfig | None = None) -> dict:
    """
    Demonstra o contrato de dados sem carregar o conteúdo dos arquivos.

    Apenas:
    - lista arquivos e subject_id derivados do nome;
    - valida colunas em exemplos mínimos ilustrativos.
    """
    if config is None:
        config = build_default_config()

    print("=" * 60)
    print("ETAPA 2 — Estrutura esperada dos arquivos")
    print("=" * 60)

    print("\nColunas obrigatórias (padrão canônico):")
    for col in REQUIRED_COLUMNS:
        role = "entrada IMU" if col in INPUT_FEATURE_COLUMNS else ("tempo" if col == TIME_COLUMN else "alvo Vicon")
        print(f"  - {col:<8} ({role})")

    print("\nModos de alvo:")
    for info in TARGET_MODE_DOCS.values():
        print(f"  [{info.mode}] {info.summary}")
        print(f"           formato y: {info.y_shape_example}")

    specs = list_expected_subject_files(config.data_dir)
    print(f"\nArquivos encontrados em {config.data_dir}: {len(specs)}")
    if specs:
        print("  Exemplos de subject_id extraído do nome:")
        for spec in specs[:3]:
            print(f"    {spec.filename} -> subject_id='{spec.subject_id}'")
        if len(specs) > 3:
            print(f"    ... (+{len(specs) - 3} arquivos)")

    # validação ilustrativa — DataFrames mínimos, não dados reais
    print("\nValidação ilustrativa (exemplos mínimos):")
    ok_canonical = validate_required_columns(make_minimal_example_dataframe())
    print(f"  canônico : {ok_canonical.message}")

    ok_aliases = validate_required_columns(make_minimal_example_with_aliases())
    print(f"  aliases  : {ok_aliases.message}")
    print(f"  mapeamento: {ok_aliases.resolved_mapping}")

    print("\nPróxima etapa: ler arquivos de fato (Etapa 3).")
    print("=" * 60)

    return {
        "required_columns": REQUIRED_COLUMNS,
        "target_mode": config.target_mode,
        "subject_file_specs": specs,
    }


if __name__ == "__main__":
    run_stage02_schema()
