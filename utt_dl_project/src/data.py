#!/usr/bin/env python3
"""Carregamento, validação, split por sujeito, baseline e janelamento."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

REQUIRED_COLUMNS = [
    "Time",
    "accX_m_s2",
    "accY_m_s2",
    "accZ_m_s2",
    "gyroX_rad_s",
    "gyroY_rad_s",
    "gyroZ_rad_s",
    "velocidade_corrigida_smart_m_s",
    "deslocamento_corrigido_smart_m",
    "vicon_esternoZ_cm",
]

INPUT_FEATURE_NAMES = [
    "time_norm",
    "accX_m_s2",
    "accY_m_s2",
    "accZ_m_s2",
    "gyroX_rad_s",
    "gyroY_rad_s",
    "gyroZ_rad_s",
    "smart_vel_cm_s",
    "smart_disp_cm",
    "smart_calibrado_cm",
]


@dataclass
class FileRecord:
    file_path: Path
    file_name: str
    subject_id: str
    n_samples: int
    duration_s: float
    df: pd.DataFrame = field(repr=False)


@dataclass
class LoadReport:
    files_ok: list[FileRecord]
    files_bad: list[dict]
    n_files: int
    n_subjects: int
    subject_ids: list[str]


@dataclass
class LinearBaseline:
    slope: float
    intercept: float

    def apply(self, smart_disp_cm: np.ndarray) -> np.ndarray:
        return self.slope * smart_disp_cm + self.intercept


@dataclass
class SubjectSplit:
    train_subjects: list[str]
    val_subjects: list[str]
    train_files: list[FileRecord]
    val_files: list[FileRecord]


def extract_subject_id(file_name: str) -> str:
    return file_name.split("_")[0]


def discover_csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*_alinhado_ml.csv"))


def validate_dataframe(df: pd.DataFrame, file_name: str) -> list[str]:
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"colunas ausentes: {missing}")
        return errors

    df_sorted = df.sort_values("Time").reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        arr = df_sorted[col].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            n_nan = int(np.sum(~np.isfinite(arr)))
            errors.append(f"coluna {col}: {n_nan} valores NaN/inf")
    return errors


def load_all_files(data_dir: Path) -> LoadReport:
    paths = discover_csv_files(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"Nenhum arquivo *_alinhado_ml.csv encontrado em {data_dir}"
        )

    files_ok: list[FileRecord] = []
    files_bad: list[dict] = []

    for path in paths:
        file_name = path.name
        subject_id = extract_subject_id(file_name)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            files_bad.append({"file": file_name, "subject_id": subject_id, "error": str(exc)})
            continue

        errors = validate_dataframe(df, file_name)
        if errors:
            files_bad.append({"file": file_name, "subject_id": subject_id, "errors": errors})
            continue

        df = df.sort_values("Time").reset_index(drop=True)
        duration = float(df["Time"].iloc[-1] - df["Time"].iloc[0]) if len(df) > 1 else 0.0
        files_ok.append(
            FileRecord(
                file_path=path,
                file_name=file_name,
                subject_id=subject_id,
                n_samples=len(df),
                duration_s=duration,
                df=df,
            )
        )

    subject_ids = sorted({r.subject_id for r in files_ok})
    return LoadReport(
        files_ok=files_ok,
        files_bad=files_bad,
        n_files=len(files_ok),
        n_subjects=len(subject_ids),
        subject_ids=subject_ids,
    )


def print_load_report(report: LoadReport) -> None:
    print(f"Arquivos lidos com sucesso: {report.n_files}")
    print(f"Sujeitos únicos          : {report.n_subjects}")
    if report.files_bad:
        print(f"\n[AVISO] Arquivos com problemas ({len(report.files_bad)}):")
        for item in report.files_bad:
            print(f"  - {item['file']}: {item.get('errors', item.get('error'))}")


def split_by_subject(
    records: list[FileRecord],
    *,
    train_ratio: float = 0.70,
    seed: int = 42,
) -> SubjectSplit:
    subject_ids = sorted({r.subject_id for r in records})
    if len(subject_ids) < 2:
        raise ValueError("São necessários pelo menos 2 sujeitos para split 70/30.")

    n_train = max(1, int(round(len(subject_ids) * train_ratio)))
    n_train = min(n_train, len(subject_ids) - 1)

    train_subjects, val_subjects = train_test_split(
        subject_ids,
        train_size=n_train,
        random_state=seed,
        shuffle=True,
    )
    train_set = set(train_subjects)
    val_set = set(val_subjects)

    train_files = [r for r in records if r.subject_id in train_set]
    val_files = [r for r in records if r.subject_id in val_set]

    if train_set & val_set:
        raise RuntimeError("Vazamento: sujeito em treino e validação simultaneamente.")

    return SubjectSplit(
        train_subjects=sorted(train_subjects),
        val_subjects=sorted(val_subjects),
        train_files=train_files,
        val_files=val_files,
    )


def save_split_csv(split: SubjectSplit, out_path: Path) -> None:
    rows = []
    for sid in split.train_subjects:
        rows.append({"subject_id": sid, "split": "train"})
    for sid in split.val_subjects:
        rows.append({"subject_id": sid, "split": "validation"})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def enrich_dataframe(df: pd.DataFrame, baseline: LinearBaseline | None = None) -> pd.DataFrame:
    out = df.copy()
    out["smart_disp_cm"] = out["deslocamento_corrigido_smart_m"] * 100.0
    out["smart_vel_cm_s"] = out["velocidade_corrigida_smart_m_s"] * 100.0
    t = out["Time"].to_numpy(dtype=float)
    t_min, t_max = t.min(), t.max()
    out["time_norm"] = (t - t_min) / max(t_max - t_min, 1e-8)
    if baseline is not None:
        out["smart_calibrado_cm"] = baseline.apply(out["smart_disp_cm"].to_numpy(dtype=float))
    return out


def fit_linear_baseline(train_files: list[FileRecord]) -> LinearBaseline:
    xs: list[float] = []
    ys: list[float] = []
    for rec in train_files:
        disp_m = rec.df["deslocamento_corrigido_smart_m"].to_numpy(dtype=float)
        smart_cm = disp_m * 100.0
        vicon = rec.df["vicon_esternoZ_cm"].to_numpy(dtype=float)
        xs.extend(smart_cm.tolist())
        ys.extend(vicon.tolist())

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        raise ValueError("Dados insuficientes para ajustar baseline linear.")

    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return LinearBaseline(slope=float(slope), intercept=float(intercept))


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = INPUT_FEATURE_NAMES
    return df[cols].to_numpy(dtype=np.float32).T  # (n_channels, T)


def iter_windows(
    arr: np.ndarray,
    window_size: int,
    stride: int,
) -> Iterator[tuple[int, np.ndarray]]:
    n = arr.shape[-1]
    if n < window_size:
        return
    for start in range(0, n - window_size + 1, stride):
        yield start, arr[..., start : start + window_size]


@dataclass
class ScalerBundle:
    feature_scaler: StandardScaler
    residual_scaler: StandardScaler | None = None

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        """x: (n_channels, T) -> scaled same shape."""
        n_ch, T = x.shape
        flat = x.T.reshape(-1, n_ch)
        scaled = self.feature_scaler.transform(flat).astype(np.float32)
        return scaled.T.reshape(n_ch, T)

    def transform_residual(self, r: np.ndarray) -> np.ndarray:
        if self.residual_scaler is None:
            return r.astype(np.float32)
        flat = r.reshape(-1, 1)
        return self.residual_scaler.transform(flat).reshape(r.shape).astype(np.float32)

    def inverse_residual(self, r_scaled: np.ndarray) -> np.ndarray:
        if self.residual_scaler is None:
            return r_scaled.astype(np.float32)
        flat = r_scaled.reshape(-1, 1)
        return self.residual_scaler.inverse_transform(flat).reshape(r_scaled.shape).astype(np.float32)


def fit_scalers(
    train_files: list[FileRecord],
    baseline: LinearBaseline,
    *,
    window_size: int,
    stride: int,
    scale_residual: bool = True,
) -> ScalerBundle:
    feature_rows: list[np.ndarray] = []
    residual_vals: list[float] = []

    for rec in train_files:
        df = enrich_dataframe(rec.df, baseline)
        x = build_feature_matrix(df)
        vicon = df["vicon_esternoZ_cm"].to_numpy(dtype=np.float32)
        cal = df["smart_calibrado_cm"].to_numpy(dtype=np.float32)
        resid = vicon - cal

        for _, xw in iter_windows(x, window_size, stride):
            feature_rows.append(xw.T)
        for _, rw in iter_windows(resid, window_size, stride):
            residual_vals.extend(rw.tolist())

    feat_mat = np.vstack(feature_rows)
    feat_scaler = StandardScaler()
    feat_scaler.fit(feat_mat)

    res_scaler = None
    if scale_residual and residual_vals:
        res_scaler = StandardScaler()
        res_scaler.fit(np.asarray(residual_vals, dtype=float).reshape(-1, 1))

    return ScalerBundle(feature_scaler=feat_scaler, residual_scaler=res_scaler)


class WindowDataset(Dataset):
    """Janelas temporais para treino/validação em modo batch."""

    def __init__(
        self,
        files: list[FileRecord],
        baseline: LinearBaseline,
        scalers: ScalerBundle,
        *,
        window_size: int = 512,
        stride: int = 128,
    ) -> None:
        self.window_size = window_size
        self.samples: list[dict] = []

        for rec in files:
            df = enrich_dataframe(rec.df, baseline)
            x = build_feature_matrix(df)
            vicon = df["vicon_esternoZ_cm"].to_numpy(dtype=np.float32)
            cal = df["smart_calibrado_cm"].to_numpy(dtype=np.float32)
            resid = vicon - cal

            for start, xw in iter_windows(x, window_size, stride):
                end = start + window_size
                self.samples.append(
                    {
                        "x": scalers.transform_features(xw),
                        "vicon": vicon[start:end],
                        "calibrado": cal[start:end],
                        "residual": scalers.transform_residual(resid[start:end]),
                        "subject_id": rec.subject_id,
                        "file_name": rec.file_name,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[idx]
        return {
            "x": torch.from_numpy(s["x"]),
            "vicon": torch.from_numpy(s["vicon"]),
            "calibrado": torch.from_numpy(s["calibrado"]),
            "residual": torch.from_numpy(s["residual"]),
            "subject_id": s["subject_id"],
            "file_name": s["file_name"],
        }


def hann_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    return np.hanning(length).astype(np.float32)
