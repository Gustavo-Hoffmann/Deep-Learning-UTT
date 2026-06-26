#!/usr/bin/env python3
"""Utilitários: device, seed, timing e diretórios de saída."""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

DeviceChoice = Literal["auto", "cpu", "cuda", "rocm", "directml", "mps"]
BackendName = Literal["cpu", "cuda", "rocm", "directml", "mps"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent


class DeviceResolutionError(RuntimeError):
    pass


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_import_directml():
    try:
        import torch_directml  # type: ignore[import-untyped]

        return torch_directml
    except ImportError:
        return None


def directml_available() -> bool:
    dml = _try_import_directml()
    if dml is None:
        return False
    try:
        _ = dml.device()
        return True
    except Exception:
        return False


def get_directml_device() -> torch.device:
    dml = _try_import_directml()
    if dml is None:
        raise DeviceResolutionError(
            "DirectML solicitado, mas torch-directml não está instalado.\n"
            "Windows + GPU AMD: pip install -r requirements-amd.txt (Python 3.11)"
        )
    return dml.device()


def is_directml_device(device: torch.device) -> bool:
    return device.type in ("privateuseone", "dml")


def torch_version_hip() -> str | None:
    return getattr(torch.version, "hip", None)


def torch_version_cuda() -> str | None:
    return getattr(torch.version, "cuda", None)


def resolve_device(request: DeviceChoice = "auto") -> tuple[torch.device, str, BackendName]:
    req = request.lower().strip()

    if req == "cpu":
        return torch.device("cpu"), "CPU (forçado)", "cpu"

    if req == "directml":
        if not directml_available():
            import sys

            py = sys.version.split()[0]
            try:
                import torch as _torch

                torch_ver = _torch.__version__
            except ImportError:
                torch_ver = "nao instalado"
            raise DeviceResolutionError(
                "DirectML indisponivel neste Python.\n"
                f"  Python atual : {py}\n"
                f"  torch atual  : {torch_ver}\n"
                "  Requer Python 3.11 e pacote torch-directml (substitui o torch CPU/CUDA):\n"
                "    pip install -r requirements-amd.txt\n"
                "  Ou apenas:\n"
                "    pip install torch-directml"
            )
        return get_directml_device(), "GPU AMD via DirectML", "directml"

    if req == "cuda":
        if not torch.cuda.is_available() or torch_version_hip() is not None:
            raise DeviceResolutionError("CUDA NVIDIA indisponível neste ambiente.")
        return torch.device("cuda"), "GPU NVIDIA/CUDA", "cuda"

    if req == "rocm":
        if not torch.cuda.is_available() or torch_version_hip() is None:
            raise DeviceResolutionError("ROCm indisponível neste ambiente.")
        return torch.device("cuda"), "GPU AMD/ROCm", "rocm"

    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise DeviceResolutionError("MPS indisponível.")
        return torch.device("mps"), "Apple MPS", "mps"

    # auto
    if torch.cuda.is_available():
        if torch_version_hip() is not None:
            return torch.device("cuda"), "GPU AMD/ROCm (auto)", "rocm"
        return torch.device("cuda"), "GPU CUDA (auto)", "cuda"
    if directml_available():
        return get_directml_device(), "GPU AMD via DirectML (auto)", "directml"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (auto)", "mps"
    return torch.device("cpu"), "CPU (auto — nenhuma GPU detectada)", "cpu"


def get_device(request: DeviceChoice = "auto") -> torch.device:
    device, _, _ = resolve_device(request)
    return device


def amp_supported(device: torch.device, backend: BackendName) -> bool:
    if backend in ("cuda", "rocm") and device.type == "cuda":
        return True
    return False


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class RunTimer:
    label: str = ""
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


@dataclass
class EpochTimer:
    total_epochs: int
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _epoch_times: list[float] = field(default_factory=list, repr=False)

    def mark_epoch_end(self) -> float:
        dur = time.perf_counter() - self._t0
        self._epoch_times.append(dur)
        self._t0 = time.perf_counter()
        return dur

    def elapsed_total(self) -> float:
        return sum(self._epoch_times)

    def eta(self, current_epoch: int) -> float | None:
        if not self._epoch_times:
            return None
        avg = sum(self._epoch_times) / len(self._epoch_times)
        remaining = max(0, self.total_epochs - current_epoch)
        return avg * remaining


def create_output_dirs(out_dir: Path) -> dict[str, Path]:
    subdirs = {
        "root": out_dir,
        "metrics": out_dir / "metrics",
        "predictions": out_dir / "predictions",
        "plots": out_dir / "plots",
        "models": out_dir / "models",
        "logs": out_dir / "logs",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def default_data_dir() -> Path:
    candidates = [
        REPO_ROOT / "Inputs_DP",
        PROJECT_ROOT / "Inputs_DP",
        Path("Inputs_DP"),
    ]
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return (REPO_ROOT / "Inputs_DP").resolve()
