#!/usr/bin/env python3
"""
Etapa 1 — Preparação inicial do ambiente
========================================
Objetivo: configurar bibliotecas, reprodutibilidade, dispositivo de
computação, parâmetros do experimento e pastas de saída.

NÃO carrega dados. NÃO treina modelo. NÃO faz split.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

DeviceChoice = Literal["auto", "cpu", "cuda", "mps"]

# Matplotlib precisa de pasta gravável para cache de fontes
_mpl_cache = Path(__file__).resolve().parent.parent / ".mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

# ---------------------------------------------------------------------------
# 1. Bibliotecas
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch

# Bibliotecas usadas nas etapas seguintes (importadas aqui para validar o ambiente)
import matplotlib

matplotlib.use("Agg")  # backend sem interface gráfica (útil em servidor/terminal)
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: F401  (etapa 6)

# ---------------------------------------------------------------------------
# 2. Reprodutibilidade
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED) -> None:
    """Fixa geradores aleatórios para resultados reproduzíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Em alguns backends, estas flags reduzem variação numérica na GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 3. Detecção de dispositivo (CPU / CUDA / MPS)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """
    Escolhe o melhor dispositivo disponível.

    Prioridade: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
  """
    device, _ = resolve_device("auto")
    return device


def resolve_device(request: DeviceChoice = "auto") -> tuple[torch.device, str]:
    """
    Resolve dispositivo PyTorch conforme solicitação do usuário.

    Retorna (device, descrição legível).
    """
    req = request.lower().strip()
    if req == "cpu":
        return torch.device("cpu"), "CPU forçado pelo usuário"
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA solicitado e disponível"
        return torch.device("cpu"), "CUDA solicitado mas indisponível — fallback CPU"
    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "MPS solicitado e disponível"
        return torch.device("cpu"), "MPS solicitado mas indisponível — fallback CPU"
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (auto)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (auto)"
    return torch.device("cpu"), "CPU (auto)"


@dataclass
class RuntimeConfig:
    """Configuração de runtime (dispositivo, threads, workers)."""

    device_request: DeviceChoice = "auto"
    device_effective: str = "cpu"
    device_description: str = ""
    cpu_full_throttle: bool = False
    num_threads: int | None = None
    num_workers: int = 0

    def to_dict(self) -> dict:
        return {
            "device_request": self.device_request,
            "device_effective": self.device_effective,
            "device_description": self.device_description,
            "cpu_full_throttle": self.cpu_full_throttle,
            "num_threads": self.num_threads,
            "num_workers": self.num_workers,
        }


def configure_cpu_threads(
    *,
    num_threads: int | None = None,
    cpu_full_throttle: bool = False,
) -> int:
    """
    Configura threads PyTorch para CPU.

    Retorna o número efetivo de threads intra-op.
    """
    if num_threads is not None:
        n = max(1, int(num_threads))
    elif cpu_full_throttle:
        n = max(1, os.cpu_count() or 1)
    else:
        n = torch.get_num_threads()

    torch.set_num_threads(n)
    interop = max(1, min(n, 4))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass
    return n


def setup_runtime(
    *,
    device: DeviceChoice = "auto",
    cpu_full_throttle: bool = False,
    num_threads: int | None = None,
    num_workers: int = 0,
) -> tuple[torch.device, RuntimeConfig]:
    """Configura dispositivo, threads e retorna RuntimeConfig."""
    dev, desc = resolve_device(device)
    if device == "cpu" or dev.type == "cpu":
        effective_threads = configure_cpu_threads(
            num_threads=num_threads,
            cpu_full_throttle=cpu_full_throttle,
        )
    else:
        effective_threads = torch.get_num_threads()

    runtime = RuntimeConfig(
        device_request=device,
        device_effective=str(dev),
        device_description=desc,
        cpu_full_throttle=cpu_full_throttle,
        num_threads=effective_threads,
        num_workers=max(0, int(num_workers)),
    )
    return dev, runtime


# ---------------------------------------------------------------------------
# 4. Configuração geral do experimento
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Parâmetros centrais do pipeline — serão expandidos nas etapas seguintes."""

    # --- caminhos ---
    project_root: Path
    data_dir: Path
    output_dir: Path

    # --- reprodutibilidade ---
    seed: int = SEED

    # --- colunas esperadas (Etapa 2 detalhará o formato) ---
    time_col: str = "time"
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ]
    )
    target_col: str = "vicon"

    # --- split por sujeito (usado a partir da Etapa 5) ---
    dev_ratio: float = 0.70  # 70% desenvolvimento / 30% teste final
    # validação interna nos 70%: LOSO (Etapa 13)

    # --- janelamento (placeholders; ajustados na Etapa 7) ---
    sampling_hz: float = 60.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0  # 50% de sobreposição

    # --- tipo de alvo ---
    target_mode: Literal["amplitude", "curve"] = "amplitude"

    # --- modelo (placeholders; Etapas 9–10) ---
    model_type: Literal["cnn1d", "tcn"] = "tcn"

    @property
    def window_samples(self) -> int:
        return int(self.sampling_hz * self.window_seconds)

    @property
    def stride_samples(self) -> int:
        return int(self.sampling_hz * self.stride_seconds)

    @property
    def n_input_channels(self) -> int:
        return len(self.feature_cols)


def build_default_config(project_root: Path | None = None) -> ExperimentConfig:
    """Cria configuração padrão apontando para a pasta de dados do projeto."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    return ExperimentConfig(
        project_root=project_root,
        data_dir=project_root / "Input_ML",
        output_dir=project_root / "outputs" / "pipeline_dl",
    )


# ---------------------------------------------------------------------------
# 5. Pastas de saída
# ---------------------------------------------------------------------------
OUTPUT_SUBDIRS: tuple[str, ...] = (
    "checkpoints",   # pesos do modelo
    "metrics",       # métricas por fold e finais
    "predictions",   # predições salvas
    "plots",         # gráficos
    "splits",        # listas de sujeitos dev/test
    "configs",       # configuração serializada
    "scalers",       # normalizadores (Etapa 6)
)


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Cria árvore de pastas de saída e retorna caminhos nomeados."""
    paths: dict[str, Path] = {"root": output_dir}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        sub = output_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        paths[name] = sub
    return paths


def save_config_snapshot(
    config: ExperimentConfig,
    paths: dict[str, Path],
    runtime: RuntimeConfig | None = None,
) -> Path:
    """Salva cópia da configuração para reprodutibilidade."""
    snapshot = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": runtime.device_effective if runtime else str(get_device()),
        "runtime": runtime.to_dict() if runtime else None,
        "config": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            "window_samples": config.window_samples,
            "stride_samples": config.stride_samples,
            "n_input_channels": config.n_input_channels,
        },
    }
    out_path = paths["configs"] / "experiment_config.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# 6. Ponto de entrada da Etapa 1 (somente setup)
# ---------------------------------------------------------------------------
def run_stage01_setup(
    project_root: Path | None = None,
    *,
    device: DeviceChoice = "auto",
    cpu_full_throttle: bool = False,
    num_threads: int | None = None,
    num_workers: int = 0,
) -> tuple[ExperimentConfig, dict[str, Path], torch.device, RuntimeConfig]:
    """
    Executa apenas a preparação inicial.

    Retorna: (config, pastas_de_saída, dispositivo, runtime).
    """
    set_seed(SEED)
    device_obj, runtime = setup_runtime(
        device=device,
        cpu_full_throttle=cpu_full_throttle,
        num_threads=num_threads,
        num_workers=num_workers,
    )
    config = build_default_config(project_root)
    paths = create_output_dirs(config.output_dir)
    save_config_snapshot(config, paths, runtime=runtime)

    print("=" * 60)
    print("ETAPA 1 — Preparação inicial do ambiente")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Dispositivo : {device_obj} ({runtime.device_description})")
    if runtime.cpu_full_throttle or runtime.num_threads:
        print(f"Threads CPU : {runtime.num_threads}")
    if runtime.num_workers:
        print(f"Workers DL  : {runtime.num_workers}")
    print(f"Dados       : {config.data_dir}  (não carregados nesta etapa)")
    print(f"Saída       : {config.output_dir}")
    print(f"Semente     : {config.seed}")
    print(f"Subpastas   : {', '.join(OUTPUT_SUBDIRS)}")
    print("=" * 60)

    return config, paths, device_obj, runtime


if __name__ == "__main__":
    run_stage01_setup()
