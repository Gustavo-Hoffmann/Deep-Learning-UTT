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

DeviceChoice = Literal["auto", "cpu", "cuda", "rocm", "directml", "mps"]
BackendName = Literal["cpu", "cuda", "rocm", "directml", "mps"]

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.preprocessing import StandardScaler  # noqa: F401

# ---------------------------------------------------------------------------
# 2. Reprodutibilidade
# ---------------------------------------------------------------------------
SEED: int = 42


def set_seed(seed: int = SEED, *, deterministic: bool = True) -> None:
    """Fixa geradores aleatórios para resultados reproduzíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def torch_version_cuda() -> str | None:
    v = getattr(torch.version, "cuda", None)
    return str(v) if v else None


def torch_version_hip() -> str | None:
    v = getattr(torch.version, "hip", None)
    return str(v) if v else None


class DeviceResolutionError(RuntimeError):
    """Erro explícito quando o dispositivo solicitado não está disponível."""


def is_rocm_build() -> bool:
    return torch.cuda.is_available() and torch_version_hip() is not None


def _try_import_directml():
    try:
        import torch_directml  # type: ignore[import-untyped]

        return torch_directml
    except ImportError:
        return None


def directml_available() -> bool:
    """True se torch-directml estiver instalado (GPU AMD/Intel no Windows via DirectX 12)."""
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
            "Windows + GPU AMD (ex.: RX 7600):\n"
            "  py -3.11 -m venv .venv\n"
            "  .venv\\Scripts\\activate\n"
            "  pip install -r requirements-amd.txt\n"
            "Nota: torch-directml requer Python 3.11 (não 3.12+)."
        )
    try:
        return dml.device()
    except Exception as exc:
        raise DeviceResolutionError(f"DirectML instalado mas falhou ao criar device: {exc}") from exc


def is_directml_device(device: torch.device) -> bool:
    return device.type in ("privateuseone", "dml")


def is_gpu_backend(backend: BackendName) -> bool:
    return backend in ("cuda", "rocm", "directml")


def is_nvidia_cuda_build() -> bool:
    return torch.cuda.is_available() and torch_version_hip() is None and torch_version_cuda() is not None


def detect_backend_from_torch() -> BackendName:
    """Infere backend a partir do PyTorch instalado (sem forçar device)."""
    if torch.cuda.is_available():
        if torch_version_hip() is not None:
            return "rocm"
        return "cuda"
    if directml_available():
        return "directml"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_name(device_index: int = 0) -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(device_index)
    except Exception:
        return None


def get_gpu_memory_total_gb(device_index: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
        return round(props.total_memory / (1024**3), 2)
    except Exception:
        return None


def device_supports_non_blocking(device: torch.device) -> bool:
    return device.type == "cuda"


def use_pin_memory(device: torch.device, backend: BackendName) -> bool:
    """pin_memory só beneficia CUDA/ROCm (host→GPU NVIDIA/AMD Linux)."""
    return device.type == "cuda" and backend in ("cuda", "rocm")


def configure_device_backend(device: torch.device, backend: BackendName, *, deterministic: bool = True) -> None:
    """Ajusta backend PyTorch conforme dispositivo."""
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    elif backend == "mps":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def default_num_workers(device: torch.device, backend: BackendName) -> int:
    """Workers padrão do DataLoader."""
    if backend == "mps":
        return 0
    cores = os.cpu_count() or 1
    if device.type == "cuda" or is_directml_device(device) or backend == "directml":
        return min(8, max(2, cores // 2))
    return min(4, max(2, cores // 4))


def configure_cpu_threads(
    *,
    num_threads: int | None = None,
    cpu_full_throttle: bool = False,
) -> tuple[int, int]:
    """
    Configura threads PyTorch para CPU AMD/Intel.

    Retorna (intra_op_threads, inter_op_threads).
    """
    cores = os.cpu_count() or 1
    if num_threads is not None:
        n = max(1, int(num_threads))
    elif cpu_full_throttle:
        n = max(1, cores - 1)
    else:
        n = torch.get_num_threads()

    interop = max(1, min(4, cores // 2)) if cpu_full_throttle else max(1, min(4, n // 2))

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        interop = torch.get_num_interop_threads()

    return n, interop


def resolve_device(request: DeviceChoice = "auto") -> tuple[torch.device, str, BackendName]:
    """
    Resolve dispositivo PyTorch conforme solicitação do usuário.

    Retorna (device, descrição legível, backend).
    ROCm usa torch.device('cuda') internamente.
    """
    req = request.lower().strip()

    if req == "cpu":
        return torch.device("cpu"), "CPU AMD/CPU forçada pelo usuário", "cpu"

    if req == "rocm":
        if not torch.cuda.is_available():
            hint = (
                " No Windows nativo use --device directml (torch-directml, Python 3.11)."
                if sys.platform == "win32"
                else ""
            )
            raise DeviceResolutionError(
                "ROCm solicitado, mas PyTorch não detectou AMD GPU/ROCm. "
                "Verifique instalação do PyTorch ROCm (Linux/WSL2)." + hint
            )
        if torch_version_hip() is None:
            raise DeviceResolutionError(
                "ROCm solicitado, mas este PyTorch não foi compilado com HIP/ROCm "
                f"(torch.version.hip={torch_version_hip()}, torch.version.cuda={torch_version_cuda()}). "
                "Instale o wheel PyTorch+ROCm ou use --device cpu."
            )
        return torch.device("cuda"), "GPU AMD/ROCm", "rocm"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise DeviceResolutionError(
                "CUDA NVIDIA solicitado, mas torch.cuda.is_available() é False. "
                "Verifique drivers NVIDIA e instalação do PyTorch CUDA."
            )
        if torch_version_hip() is not None:
            raise DeviceResolutionError(
                "Este PyTorch parece ser ROCm (torch.version.hip definido), não CUDA NVIDIA. "
                "Use --device rocm ou --device auto."
            )
        return torch.device("cuda"), "GPU NVIDIA/CUDA", "cuda"

    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise DeviceResolutionError(
                "MPS solicitado, mas não está disponível nesta máquina."
            )
        return torch.device("mps"), "Apple MPS", "mps"

    if req == "directml":
        if not directml_available():
            raise DeviceResolutionError(
                "DirectML solicitado, mas torch-directml não está disponível.\n"
                "Instale com Python 3.11: pip install torch-directml\n"
                f"Python atual: {sys.version.split()[0]}"
            )
        return get_directml_device(), "GPU AMD via DirectML (DirectX 12)", "directml"

    # auto — prioridade: CUDA NVIDIA > ROCm > DirectML (Windows AMD) > MPS > CPU
    if torch.cuda.is_available():
        if torch_version_hip() is not None:
            return torch.device("cuda"), "GPU AMD/ROCm detectada", "rocm"
        if torch_version_cuda() is not None:
            return torch.device("cuda"), "GPU NVIDIA/CUDA detectada", "cuda"
        return torch.device("cuda"), "GPU CUDA detectada", "cuda"

    if directml_available():
        return get_directml_device(), "GPU AMD detectada via DirectML (DirectX 12)", "directml"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (auto)", "mps"

    return torch.device("cpu"), "Nenhuma GPU disponível no PyTorch; usando CPU", "cpu"


def get_device() -> torch.device:
    device, _, _ = resolve_device("auto")
    return device


def print_directml_diagnostics(device: torch.device) -> None:
    print(f"  DirectML device    : {device}")
    print(f"  torch-directml     : instalado")
    if sys.platform == "win32":
        print("  GPU esperada       : AMD Radeon (ex.: RX 7600) via DirectX 12")


def print_gpu_diagnostics(device: torch.device, backend: BackendName) -> None:
    """Diagnóstico inicial de GPU (CUDA/ROCm)."""
    if device.type != "cuda":
        return
    name = get_gpu_name()
    mem = get_gpu_memory_total_gb()
    print(f"  GPU nome           : {name or 'N/A'}")
    print(f"  torch.version.hip  : {torch_version_hip() or 'None'}")
    print(f"  torch.version.cuda : {torch_version_cuda() or 'None'}")
    print(f"  Backend efetivo    : {backend}")
    if mem is not None:
        print(f"  Memória GPU total  : {mem} GB")
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"  Compute capability : {props.major}.{props.minor}")
        print(f"  Multiprocessors    : {props.multi_processor_count}")
    except Exception:
        pass


def print_cpu_diagnostics() -> None:
    """Diagnóstico de CPU."""
    print(f"  os.cpu_count()              : {os.cpu_count()}")
    print(f"  torch.get_num_threads()     : {torch.get_num_threads()}")
    try:
        print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    except RuntimeError:
        print("  torch.get_num_interop_threads(): N/A")


@dataclass
class RuntimeConfig:
    """Configuração de runtime (dispositivo, threads, workers)."""

    device_request: DeviceChoice = "auto"
    device_effective: str = "cpu"
    backend: BackendName = "cpu"
    device_description: str = ""
    cpu_full_throttle: bool = False
    num_threads: int | None = None
    num_interop_threads: int | None = None
    num_workers: int = 0
    pin_memory: bool = False
    gpu_name: str | None = None
    gpu_memory_total_gb: float | None = None
    torch_version: str = ""
    torch_version_cuda: str | None = None
    torch_version_hip: str | None = None

    def to_dict(self) -> dict:
        return {
            "device_request": self.device_request,
            "device_effective": self.device_effective,
            "backend": self.backend,
            "device_description": self.device_description,
            "cpu_full_throttle": self.cpu_full_throttle,
            "num_threads": self.num_threads,
            "num_interop_threads": self.num_interop_threads,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "gpu_name": self.gpu_name,
            "gpu_memory_total_gb": self.gpu_memory_total_gb,
            "torch_version": self.torch_version,
            "torch_version_cuda": self.torch_version_cuda,
            "torch_version_hip": self.torch_version_hip,
        }


def setup_runtime(
    *,
    device: DeviceChoice = "auto",
    cpu_full_throttle: bool = False,
    num_threads: int | None = None,
    num_workers: int | None = None,
    auto_workers: bool = True,
) -> tuple[torch.device, RuntimeConfig]:
    """Configura dispositivo, threads e retorna RuntimeConfig."""
    dev, desc, backend = resolve_device(device)
    configure_device_backend(dev, backend, deterministic=True)

    num_interop: int | None = None
    if device == "cpu" or dev.type == "cpu" or backend == "cpu":
        effective_threads, num_interop = configure_cpu_threads(
            num_threads=num_threads,
            cpu_full_throttle=cpu_full_throttle,
        )
    else:
        effective_threads = torch.get_num_threads()
        try:
            num_interop = torch.get_num_interop_threads()
        except RuntimeError:
            num_interop = None

    if num_workers is None:
        effective_workers = default_num_workers(dev, backend) if auto_workers else 0
    else:
        effective_workers = max(0, int(num_workers))

    pin_mem = use_pin_memory(dev, backend)

    runtime = RuntimeConfig(
        device_request=device,
        device_effective=str(dev),
        backend=backend,
        device_description=desc,
        cpu_full_throttle=cpu_full_throttle,
        num_threads=effective_threads,
        num_interop_threads=num_interop,
        num_workers=effective_workers,
        pin_memory=pin_mem,
        gpu_name=get_gpu_name() if dev.type == "cuda" else ("DirectML" if backend == "directml" else None),
        gpu_memory_total_gb=get_gpu_memory_total_gb() if dev.type == "cuda" else None,
        torch_version=torch.__version__,
        torch_version_cuda=torch_version_cuda(),
        torch_version_hip=torch_version_hip(),
    )
    return dev, runtime


def print_runtime_banner(runtime: RuntimeConfig, device: torch.device) -> None:
    """Imprime diagnóstico de hardware no início do run."""
    print("=" * 60)
    print("HARDWARE / RUNTIME")
    print("=" * 60)
    print(f"  Solicitado           : {runtime.device_request}")
    print(f"  Dispositivo PyTorch  : {runtime.device_effective}")
    print(f"  Backend              : {runtime.backend}")
    print(f"  Descrição            : {runtime.device_description}")
    print(f"  PyTorch              : {runtime.torch_version}")

    if runtime.backend in ("cuda", "rocm"):
        print_gpu_diagnostics(device, runtime.backend)
    elif runtime.backend == "directml":
        print_directml_diagnostics(device)
    if runtime.backend == "cpu" or device.type == "cpu":
        print_cpu_diagnostics()

    print(f"  DataLoader workers   : {runtime.num_workers}")
    print(f"  pin_memory           : {runtime.pin_memory}")
    if runtime.cpu_full_throttle:
        print(f"  CPU full throttle    : sim (threads={runtime.num_threads}, interop={runtime.num_interop_threads})")
    elif runtime.num_threads:
        print(f"  CPU threads          : {runtime.num_threads}")

    if sys.platform == "win32" and runtime.num_workers > 0:
        print("  Dica Windows         : se travar, rode com --num-workers 0")
    print("=" * 60)


def check_hardware(
    *,
    device: DeviceChoice = "auto",
    cpu_full_throttle: bool = False,
    num_threads: int | None = None,
    num_workers: int | None = None,
    auto_workers: bool = True,
) -> int:
    """
    Diagnóstico de hardware sem executar o pipeline.

    Retorna código de saída (0 = ok).
    """
    print("=" * 60)
    print("CHECK HARDWARE — diagnóstico")
    print("=" * 60)
    print(f"Python               : {sys.version.split()[0]} ({sys.platform})")
    print(f"PyTorch              : {torch.__version__}")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"torch.version.cuda   : {torch_version_cuda() or 'None'}")
    print(f"torch.version.hip    : {torch_version_hip() or 'None'}")
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"torch.backends.mps   : {mps_avail}")
    print(f"torch-directml       : {directml_available()}")
    print(f"Backend detectado    : {detect_backend_from_torch()}")
    print(f"os.cpu_count()       : {os.cpu_count()}")

    try:
        dev, desc, backend = resolve_device(device)
        _, runtime = setup_runtime(
            device=device,
            cpu_full_throttle=cpu_full_throttle,
            num_threads=num_threads,
            num_workers=num_workers,
            auto_workers=auto_workers,
        )
        print(f"\nDevice (--device {device}):")
        print(f"  Escolhido            : {dev} ({desc})")
        print(f"  Backend              : {backend}")
        print_runtime_banner(runtime, dev)
        return 0
    except DeviceResolutionError as exc:
        print(f"\nERRO: {exc}")
        return 1


# ---------------------------------------------------------------------------
# 4. Configuração geral do experimento
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Parâmetros centrais do pipeline — serão expandidos nas etapas seguintes."""

    project_root: Path
    data_dir: Path
    output_dir: Path
    seed: int = SEED
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
    dev_ratio: float = 0.70
    sampling_hz: float = 60.0
    window_seconds: float = 2.0
    stride_seconds: float = 1.0
    target_mode: Literal["amplitude", "curve"] = "curve"
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
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return ExperimentConfig(
        project_root=project_root,
        data_dir=project_root / "Input_ML",
        output_dir=project_root / "outputs" / "pipeline_dl",
    )


OUTPUT_SUBDIRS: tuple[str, ...] = (
    "checkpoints",
    "metrics",
    "predictions",
    "plots",
    "splits",
    "configs",
    "scalers",
)


def create_output_dirs(output_dir: Path) -> dict[str, Path]:
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


def run_stage01_setup(
    project_root: Path | None = None,
    *,
    device: DeviceChoice = "auto",
    cpu_full_throttle: bool = False,
    num_threads: int | None = None,
    num_workers: int | None = None,
    auto_workers: bool = True,
) -> tuple[ExperimentConfig, dict[str, Path], torch.device, RuntimeConfig]:
    set_seed(SEED)
    device_obj, runtime = setup_runtime(
        device=device,
        cpu_full_throttle=cpu_full_throttle,
        num_threads=num_threads,
        num_workers=num_workers,
        auto_workers=auto_workers,
    )
    config = build_default_config(project_root)
    paths = create_output_dirs(config.output_dir)
    save_config_snapshot(config, paths, runtime=runtime)

    print("=" * 60)
    print("ETAPA 1 — Preparação inicial do ambiente")
    print("=" * 60)
    print_runtime_banner(runtime, device_obj)
    print(f"Dados       : {config.data_dir}")
    print(f"Saída       : {config.output_dir}")
    print(f"Semente     : {config.seed}")
    print("=" * 60)

    return config, paths, device_obj, runtime


if __name__ == "__main__":
    run_stage01_setup()
