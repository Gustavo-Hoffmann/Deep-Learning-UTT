#!/usr/bin/env python3
"""
Etapa 8 — Dataset e DataLoader no PyTorch
=========================================
Encapsula janelas (Etapa 7) em Dataset e DataLoader para treino/validação.

NÃO treina modelo. NÃO mistura sujeitos entre conjuntos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs, get_device
from etapa02_schema import TargetMode
from etapa03_load import load_all_subjects
from etapa05_split import apply_split_to_dataset, load_subject_split
from etapa06_normalize import normalize_loso_fold
from etapa07_windows import (
    WindowBatch,
    assert_subject_isolation,
    create_windows_from_dataset,
)


# ---------------------------------------------------------------------------
# 1. Dataset PyTorch
# ---------------------------------------------------------------------------
class IMUWindowDataset(Dataset):
    """
    Dataset de janelas IMU -> alvo Vicon.

    Cada item devolve:
        x         : Tensor (n_canais, tamanho_janela)
        y         : Tensor escalar () em modo amplitude
                    ou (tamanho_janela,) em modo curve
        subject_id: str
    """

    def __init__(self, batch: WindowBatch) -> None:
        self.batch = batch
        self.target_mode: TargetMode = batch.target_mode

        if batch.n_windows == 0:
            raise ValueError("WindowBatch vazio — não é possível criar Dataset.")

    def __len__(self) -> int:
        return self.batch.n_windows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        x = torch.from_numpy(self.batch.X[index])          # (canais, tempo)
        y_np = self.batch.y[index]

        if self.target_mode == "amplitude":
            y = torch.tensor(float(y_np), dtype=torch.float32)
        else:
            y = torch.from_numpy(np.asarray(y_np, dtype=np.float32))

        return {
            "x": x,
            "y": y,
            "subject_id": str(self.batch.subject_ids[index]),
            "window_start_idx": int(self.batch.window_start_idx[index]),
            "window_start_time": float(self.batch.window_start_time[index]),
        }

    @classmethod
    def from_window_batch(cls, batch: WindowBatch) -> IMUWindowDataset:
        return cls(batch)

    @property
    def subject_ids(self) -> list[str]:
        return sorted(set(self.batch.subject_ids.tolist()))


# ---------------------------------------------------------------------------
# 2. Collation (empilhar um batch)
# ---------------------------------------------------------------------------
def collate_imu_windows(items: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """
    Combina amostras individuais num mini-batch.

    Saída:
        x  : (batch, canais, tempo)
        y  : (batch,) ou (batch, tempo) em modo curve
    """
    x = torch.stack([item["x"] for item in items], dim=0)
    y = torch.stack([item["y"] for item in items], dim=0)
    subject_ids = [str(item["subject_id"]) for item in items]
    window_start_time = torch.tensor(
        [float(item["window_start_time"]) for item in items],
        dtype=torch.float32,
    )

    return {
        "x": x,
        "y": y,
        "subject_ids": subject_ids,
        "window_start_time": window_start_time,
    }


# ---------------------------------------------------------------------------
# 3. DataLoaders por conjunto (treino / val / teste)
# ---------------------------------------------------------------------------
@dataclass
class FoldDataLoaders:
    """DataLoaders de um fold (ex.: LOSO)."""

    train: DataLoader
    val: DataLoader
    train_dataset: IMUWindowDataset
    val_dataset: IMUWindowDataset
    batch_size: int

    def inspect_one_batch(self, split: str = "train") -> dict[str, torch.Tensor | list[str]]:
        loader = self.train if split == "train" else self.val
        return next(iter(loader))


def create_dataloader(
    dataset: IMUWindowDataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """
    Cria DataLoader a partir do Dataset.

    shuffle=True  -> apenas treino (mistura janelas dentro do conjunto)
    shuffle=False -> validação e teste (ordem fixa)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_imu_windows,
    )


def create_fold_dataloaders(
    train_batch: WindowBatch,
    val_batch: WindowBatch,
    batch_size: int = 32,
    num_workers: int = 0,
    drop_last_train: bool = False,
) -> FoldDataLoaders:
    """
    Monta DataLoaders de treino e validação para um fold.

    Pré-condição: train_batch e val_batch vêm de sujeitos disjuntos.
    """
    assert_subject_isolation(train_batch, val_batch)

    train_ds = IMUWindowDataset.from_window_batch(train_batch)
    val_ds = IMUWindowDataset.from_window_batch(val_batch)

    train_loader = create_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # só mistura janelas DENTRO do treino
        num_workers=num_workers,
        drop_last=drop_last_train,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          # validação: ordem determinística
        num_workers=num_workers,
        drop_last=False,
    )

    return FoldDataLoaders(
        train=train_loader,
        val=val_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# 4. Inspeção didática de batches
# ---------------------------------------------------------------------------
def describe_batch(batch: dict[str, torch.Tensor | list[str]], split_name: str) -> None:
    """Imprime formas e tipos de um mini-batch."""
    x = batch["x"]
    y = batch["y"]
    print(f"  [{split_name}]")
    print(f"    x.shape : {tuple(x.shape)}  (batch, canais, tempo)")
    print(f"    x.dtype : {x.dtype}")
    print(f"    y.shape : {tuple(y.shape)}")
    print(f"    y.dtype : {y.dtype}")
    print(f"    subject_ids (primeiros 5): {batch['subject_ids'][:5]}")


def iterate_batches(
    loader: DataLoader,
    max_batches: int = 1,
) -> Iterator[dict[str, torch.Tensor | list[str]]]:
    """Itera no máximo ``max_batches`` lotes (sem treinar)."""
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch


def verify_loader_subject_pool(
    loader: DataLoader,
    allowed_subjects: set[str],
    split_name: str,
) -> None:
    """Confere se todas as janelas do loader pertencem ao pool permitido."""
    seen: set[str] = set()
    for batch in loader:
        for sid in batch["subject_ids"]:
            seen.add(sid)
            if sid not in allowed_subjects:
                raise ValueError(
                    f"Sujeito {sid!r} inesperado no loader '{split_name}'. "
                    f"Permitidos: {sorted(allowed_subjects)}"
                )


# ---------------------------------------------------------------------------
# 5. Relatório legível
# ---------------------------------------------------------------------------
def print_dataloader_report(
    loaders: FoldDataLoaders,
    train_batch: WindowBatch,
    val_batch: WindowBatch,
    val_subject_id: str,
    device: torch.device,
) -> None:
    print("=" * 60)
    print("ETAPA 8 — Dataset e DataLoader no PyTorch")
    print("=" * 60)
    print(f"Dispositivo alvo   : {device}")
    print(f"Batch size         : {loaders.batch_size}")
    print(f"Treino — amostras  : {len(loaders.train_dataset)}")
    print(f"Treino — batches   : {len(loaders.train)}")
    print(f"Val    — amostras  : {len(loaders.val_dataset)}")
    print(f"Val    — batches   : {len(loaders.val)}")

    print(f"\nSujeitos no treino : {loaders.train_dataset.subject_ids}")
    print(f"Sujeitos na val    : {loaders.val_dataset.subject_ids}")

    train_allowed = set(loaders.train_dataset.subject_ids)
    val_allowed = set(loaders.val_dataset.subject_ids)
    verify_loader_subject_pool(loaders.train, train_allowed, "train")
    verify_loader_subject_pool(loaders.val, val_allowed, "val")
    print("\n✓ Loaders respeitam pools de sujeitos (sem mistura treino/val).")

    print("\nExemplo de mini-batch:")
    train_mini = loaders.inspect_one_batch("train")
    val_mini = loaders.inspect_one_batch("val")
    describe_batch(train_mini, "treino")
    describe_batch(val_mini, "validação")

    # demonstração de envio para device (sem treino)
    x_train = train_mini["x"].to(device)
    y_train = train_mini["y"].to(device)
    print(f"\n  Tensores no device '{device}':")
    print(f"    x_train : {tuple(x_train.shape)}, {x_train.dtype}")
    print(f"    y_train : {tuple(y_train.shape)}, {y_train.dtype}")

    print("\nRegras de shuffle:")
    print("  - treino     : shuffle=True  (janelas do mesmo conjunto)")
    print("  - validação  : shuffle=False (reprodutível)")
    print("  - teste final: shuffle=False (Etapa 15)")

    print("\nPróxima etapa: modelo CNN 1D inicial (Etapa 9).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Ponto de entrada da Etapa 8
# ---------------------------------------------------------------------------
def run_stage08_dataloader(
    config: ExperimentConfig | None = None,
    example_val_subject_id: str = "02",
    batch_size: int = 32,
) -> FoldDataLoaders:
    """
    Monta Dataset e DataLoaders para um fold LOSO de desenvolvimento.

    Não usa sujeitos de teste final (30%).
    """
    if config is None:
        config = build_default_config()

    paths = create_output_dirs(config.output_dir)
    device = get_device()

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
        target_mode=config.target_mode,
    )
    val_batch = create_windows_from_dataset(
        fold.val_normalized,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=config.target_mode,
    )

    loaders = create_fold_dataloaders(
        train_batch,
        val_batch,
        batch_size=batch_size,
    )

    print_dataloader_report(
        loaders,
        train_batch,
        val_batch,
        example_val_subject_id,
        device,
    )
    return loaders


if __name__ == "__main__":
    run_stage08_dataloader()
