#!/usr/bin/env python3
"""
Etapa 5 — Separação externa 70/30 por sujeito
==============================================
Divide os sujeitos em dois grupos disjuntos:
  - desenvolvimento (70%): treino + LOSO interno
  - teste final (30%): intocado até a Etapa 15

NÃO normaliza. NÃO cria janelas. NÃO treina modelo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from etapa01_setup import SEED, ExperimentConfig, build_default_config, create_output_dirs
from etapa03_load import LoadedDataset, LoadSummary, load_all_subjects


# ---------------------------------------------------------------------------
# 1. Estrutura do split externo
# ---------------------------------------------------------------------------
@dataclass
class SubjectSplit:
    """
    Partição externa por sujeito.

    dev_subject_ids  -> usados nas Etapas 6–14 (LOSO, hiperparâmetros)
    test_subject_ids -> reservados; só Etapa 15 em diante
    """

    dev_subject_ids: list[str]
    test_subject_ids: list[str]
    seed: int
    dev_ratio: float
    n_subjects_total: int

    @property
    def n_dev(self) -> int:
        return len(self.dev_subject_ids)

    @property
    def n_test(self) -> int:
        return len(self.test_subject_ids)

    @property
    def dev_fraction(self) -> float:
        return self.n_dev / self.n_subjects_total if self.n_subjects_total else 0.0

    @property
    def test_fraction(self) -> float:
        return self.n_test / self.n_subjects_total if self.n_subjects_total else 0.0

    def validate(self) -> None:
        """Garante integridade do split."""
        dev_set = set(self.dev_subject_ids)
        test_set = set(self.test_subject_ids)

        overlap = dev_set & test_set
        if overlap:
            raise ValueError(f"Sujeitos em dev E teste: {sorted(overlap)}")

        if len(dev_set) != len(self.dev_subject_ids):
            raise ValueError("IDs duplicados no grupo de desenvolvimento.")

        if len(test_set) != len(self.test_subject_ids):
            raise ValueError("IDs duplicados no grupo de teste final.")

        union = dev_set | test_set
        if len(union) != self.n_subjects_total:
            raise ValueError(
                f"Esperado {self.n_subjects_total} sujeitos únicos, "
                f"mas dev+test cobrem {len(union)}."
            )

        if self.n_dev < 1 or self.n_test < 1:
            raise ValueError(
                "Split inválido: é necessário pelo menos 1 sujeito em dev e 1 em teste."
            )


# ---------------------------------------------------------------------------
# 2. Função de split por sujeito
# ---------------------------------------------------------------------------
def split_subjects_external(
    subject_ids: list[str],
    dev_ratio: float = 0.70,
    seed: int = SEED,
) -> SubjectSplit:
    """
    Divide sujeitos em desenvolvimento e teste final.

    O split é feito sobre **IDs de sujeito**, nunca sobre janelas ou linhas.

    Parameters
    ----------
    subject_ids :
        Lista de identificadores únicos (ex.: ['01', '02', ...]).
    dev_ratio :
        Fração alvo para desenvolvimento (padrão 0.70).
    seed :
        Semente para embaralhar sujeitos de forma reproduzível.

    Returns
    -------
    SubjectSplit com listas ordenadas de IDs.
    """
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio deve estar entre 0 e 1 (recebido: {dev_ratio})")

    unique_ids = sorted(set(str(s) for s in subject_ids))
    n_total = len(unique_ids)

    if n_total < 2:
        raise ValueError(
            f"São necessários pelo menos 2 sujeitos para split 70/30 (encontrados: {n_total})."
        )

    # embaralha com RNG determinístico (sem depender de numpy)
    import random

    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    # arredonda para aproximar 70/30
    n_dev = max(1, min(n_total - 1, round(n_total * dev_ratio)))
    dev_ids = sorted(shuffled[:n_dev])
    test_ids = sorted(shuffled[n_dev:])

    split = SubjectSplit(
        dev_subject_ids=dev_ids,
        test_subject_ids=test_ids,
        seed=seed,
        dev_ratio=dev_ratio,
        n_subjects_total=n_total,
    )
    split.validate()
    return split


# ---------------------------------------------------------------------------
# 3. Subconjuntos do LoadedDataset (sem janelas, sem normalização)
# ---------------------------------------------------------------------------
def subset_dataset_by_subjects(
    dataset: LoadedDataset,
    subject_ids: list[str],
    group_name: str,
) -> LoadedDataset:
    """
    Retorna um LoadedDataset contendo apenas os sujeitos indicados.

    Útil para isolar dev ou teste sem misturar dados.
    """
    missing = [sid for sid in subject_ids if sid not in dataset.subjects]
    if missing:
        raise KeyError(f"Sujeitos não encontrados no dataset ({group_name}): {missing}")

    subjects = {sid: dataset.subjects[sid] for sid in subject_ids}
    ids = sorted(subjects.keys())
    row_counts = [rec.n_rows for rec in subjects.values()]

    summary = LoadSummary(
        n_files_found=len(subjects),
        n_subjects_loaded=len(subjects),
        n_rows_total=sum(row_counts),
        n_rows_min=min(row_counts),
        n_rows_max=max(row_counts),
        n_rows_median=float(pd.Series(row_counts).median()),
        subjects_with_missing=[sid for sid, rec in subjects.items() if rec.missing_total > 0],
        duplicate_subject_ids=[],
        subject_ids=ids,
    )

    return LoadedDataset(subjects=subjects, summary=summary, data_dir=dataset.data_dir)


def apply_split_to_dataset(
    dataset: LoadedDataset,
    split: SubjectSplit,
) -> tuple[LoadedDataset, LoadedDataset]:
    """Separa o dataset carregado em dev e teste final."""
    split.validate()
    dev = subset_dataset_by_subjects(dataset, split.dev_subject_ids, "dev")
    test = subset_dataset_by_subjects(dataset, split.test_subject_ids, "test")
    return dev, test


# ---------------------------------------------------------------------------
# 4. Persistência do split (reprodutibilidade)
# ---------------------------------------------------------------------------
def save_subject_split(split: SubjectSplit, splits_dir: Path) -> dict[str, Path]:
    """
    Salva listas de sujeitos em JSON e TXT.

    O teste final fica registrado aqui e não deve ser alterado depois.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "description": "Split externo 70/30 por sujeito. Teste final intocado até Etapa 15.",
        "seed": split.seed,
        "dev_ratio_target": split.dev_ratio,
        "n_subjects_total": split.n_subjects_total,
        "n_dev": split.n_dev,
        "n_test": split.n_test,
        "dev_fraction_actual": round(split.dev_fraction, 4),
        "test_fraction_actual": round(split.test_fraction, 4),
        "dev_subject_ids": split.dev_subject_ids,
        "test_subject_ids": split.test_subject_ids,
    }

    json_path = splits_dir / "external_split_70_30.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    dev_txt = splits_dir / "dev_subject_ids.txt"
    test_txt = splits_dir / "test_subject_ids.txt"
    dev_txt.write_text("\n".join(split.dev_subject_ids) + "\n", encoding="utf-8")
    test_txt.write_text("\n".join(split.test_subject_ids) + "\n", encoding="utf-8")

    return {"json": json_path, "dev_txt": dev_txt, "test_txt": test_txt}


def load_subject_split(splits_dir: Path | str) -> SubjectSplit:
    """Carrega split previamente salvo."""
    json_path = Path(splits_dir) / "external_split_70_30.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Split não encontrado: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    split = SubjectSplit(
        dev_subject_ids=[str(s) for s in payload["dev_subject_ids"]],
        test_subject_ids=[str(s) for s in payload["test_subject_ids"]],
        seed=int(payload["seed"]),
        dev_ratio=float(payload["dev_ratio_target"]),
        n_subjects_total=int(payload["n_subjects_total"]),
    )
    split.validate()
    return split


# ---------------------------------------------------------------------------
# 5. Relatório legível
# ---------------------------------------------------------------------------
def print_split_report(split: SubjectSplit, saved_paths: dict[str, Path] | None = None) -> None:
    print("=" * 60)
    print("ETAPA 5 — Separação externa 70/30 por sujeito")
    print("=" * 60)
    print(f"Total de sujeitos : {split.n_subjects_total}")
    print(f"Semente           : {split.seed}")
    print(f"Alvo dev_ratio    : {split.dev_ratio:.0%}")
    print(
        f"Desenvolvimento   : {split.n_dev} sujeitos ({split.dev_fraction:.1%}) "
        f"-> {split.dev_subject_ids}"
    )
    print(
        f"Teste final       : {split.n_test} sujeitos ({split.test_fraction:.1%}) "
        f"-> {split.test_subject_ids}"
    )
    print("\n✓ Nenhum sujeito aparece nos dois grupos.")

    if saved_paths:
        print("\nArquivos salvos:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")

    print("\nRegras a partir daqui:")
    print("  - LOSO, normalização e hiperparâmetros: APENAS nos sujeitos de desenvolvimento.")
    print("  - Teste final: intocado até a Etapa 15.")
    print("\nPróxima etapa: normalização sem vazamento (Etapa 6).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Ponto de entrada da Etapa 5
# ---------------------------------------------------------------------------
def run_stage05_split(
    config: ExperimentConfig | None = None,
    dataset: LoadedDataset | None = None,
) -> tuple[SubjectSplit, LoadedDataset, LoadedDataset, dict[str, Path]]:
    """
    Cria split externo 70/30, salva listas e retorna datasets dev/teste.

    Não normaliza. Não cria janelas.
    """
    if config is None:
        config = build_default_config()

    paths = create_output_dirs(config.output_dir)

    if dataset is None:
        dataset = load_all_subjects(config.data_dir)

    split = split_subjects_external(
        subject_ids=dataset.get_subject_ids(),
        dev_ratio=config.dev_ratio,
        seed=config.seed,
    )

    saved = save_subject_split(split, paths["splits"])
    dev_dataset, test_dataset = apply_split_to_dataset(dataset, split)

    print_split_report(split, saved_paths=saved)
    return split, dev_dataset, test_dataset, saved


if __name__ == "__main__":
    run_stage05_split()
