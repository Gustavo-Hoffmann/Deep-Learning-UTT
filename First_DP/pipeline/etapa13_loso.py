#!/usr/bin/env python3
"""
Etapa 13 — LOSO completo nos 70% de desenvolvimento
===================================================
Leave-One-Subject-Out: para cada sujeito dev, treina nos demais e valida
no sujeito de fora. Normalização e janelas são refeitas a cada fold.

NÃO usa teste final (30%). NÃO ajusta hiperparâmetros com teste.

Como rodar no futuro
--------------------
    cd /Users/Rodacki/Desktop/Hoffmann/UTT
    .venv/bin/python pipeline/etapa13_loso.py

Opções úteis:
    --verbose          log de épocas por fold
    --plots            salva curva de loss de cada fold
    --model tcn        ou cnn1d
    --skip-existing    retoma folds cujo checkpoint já existe (run interrompido)

Tempo estimado: ~1–2 min/fold em CPU → ~20–40 min para 19 sujeitos dev.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from timing_utils import RunTimer, format_duration
from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs, get_device
from etapa03_load import LoadedDataset
from curve_utils import curve_window_mae, predictions_to_long_rows
from etapa11_loss_metrics import TrainingConfig, compute_regression_metrics, metrics_to_original_scale
from etapa12_train_fold import FoldTrainingResult, prepare_loso_fold, train_single_fold


# ---------------------------------------------------------------------------
# 1. Resultados agregados do LOSO
# ---------------------------------------------------------------------------
@dataclass
class SubjectFoldMetrics:
    val_subject_id: str
    n_train_windows: int
    n_val_windows: int
    best_epoch: int
    best_val_loss: float
    mae: float
    rmse: float
    rms: float
    r2: float
    bias: float
    mape_pct: float


@dataclass
class LOSOSummary:
    n_folds: int
    dev_subject_ids: list[str]
    per_subject: list[SubjectFoldMetrics]
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    mean_bias: float
    std_bias: float


def _metrics_from_result(result: FoldTrainingResult, n_train: int, n_val: int) -> SubjectFoldMetrics:
    m = result.best_val_metrics
    return SubjectFoldMetrics(
        val_subject_id=result.val_subject_id,
        n_train_windows=n_train,
        n_val_windows=n_val,
        best_epoch=result.history.best_epoch,
        best_val_loss=result.history.best_val_loss,
        mae=m.mae,
        rmse=m.rmse,
        rms=m.rms,
        r2=m.r2,
        bias=m.bias,
        mape_pct=m.mape_pct,
    )


def aggregate_loso_metrics(per_subject: list[SubjectFoldMetrics]) -> LOSOSummary:
    """Calcula média ± desvio-padrão das métricas entre folds."""

    def col(name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in per_subject], dtype=float)

    return LOSOSummary(
        n_folds=len(per_subject),
        dev_subject_ids=[s.val_subject_id for s in per_subject],
        per_subject=per_subject,
        mean_mae=float(col("mae").mean()),
        std_mae=float(col("mae").std(ddof=1)) if len(per_subject) > 1 else 0.0,
        mean_rmse=float(col("rmse").mean()),
        std_rmse=float(col("rmse").std(ddof=1)) if len(per_subject) > 1 else 0.0,
        mean_r2=float(col("r2").mean()),
        std_r2=float(col("r2").std(ddof=1)) if len(per_subject) > 1 else 0.0,
        mean_bias=float(col("bias").mean()),
        std_bias=float(col("bias").std(ddof=1)) if len(per_subject) > 1 else 0.0,
    )


# ---------------------------------------------------------------------------
# 2. LOSO sobre grupo de desenvolvimento
# ---------------------------------------------------------------------------
def load_dev_dataset(config: ExperimentConfig) -> LoadedDataset:
    """Carrega apenas sujeitos de desenvolvimento (70%)."""
    from etapa03_load import load_all_subjects
    from etapa05_split import apply_split_to_dataset, load_subject_split

    paths = create_output_dirs(config.output_dir)
    dataset = load_all_subjects(config.data_dir)
    split = load_subject_split(paths["splits"])
    dev_dataset, _test_dataset = apply_split_to_dataset(dataset, split)
    return dev_dataset


def _fold_checkpoint_path(checkpoints_dir: Path, val_subject_id: str, prefix: str = "etapa13") -> Path:
    return checkpoints_dir / f"{prefix}_loso_val_{val_subject_id}_best.pt"


def _fold_metrics_path(metrics_dir: Path, val_subject_id: str, prefix: str = "etapa13") -> Path:
    return metrics_dir / f"{prefix}_loso_val_{val_subject_id}_metrics.json"


def load_existing_fold_metrics(
    metrics_dir: Path,
    val_subject_id: str,
    n_train: int,
    n_val: int,
    prefix: str = "etapa13",
) -> SubjectFoldMetrics | None:
    """Carrega métricas de um fold já treinado (para retomar run interrompido)."""
    path = _fold_metrics_path(metrics_dir, val_subject_id, prefix=prefix)
    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    m = payload["best_val_metrics"]
    return SubjectFoldMetrics(
        val_subject_id=val_subject_id,
        n_train_windows=n_train,
        n_val_windows=n_val,
        best_epoch=int(payload["best_epoch"]),
        best_val_loss=float(payload["best_val_loss"]),
        mae=float(m["mae"]),
        rmse=float(m["rmse"]),
        rms=float(m.get("rms", m["rmse"])),
        r2=float(m["r2"]),
        bias=float(m["bias"]),
        mape_pct=float(m["mape_pct"]),
    )


def run_loso_on_dev(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    dev_dataset: LoadedDataset | None = None,
    subject_ids: list[str] | None = None,
    verbose: bool = False,
    save_individual_plots: bool = False,
    skip_existing: bool = False,
    artifact_prefix: str = "etapa13",
    device: torch.device | None = None,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    pruning_min_folds: int = 5,
    global_best_val_mae_cm: float | None = None,
) -> tuple[list[FoldTrainingResult], LOSOSummary, bool]:
    """
    Executa LOSO completo nos sujeitos de desenvolvimento.

    Para cada fold:
      1. normaliza com treino do fold (sem vazamento)
      2. cria janelas separadas treino/val
      3. treina TCN/CNN com early stopping
      4. salva métricas do sujeito de validação
    """
    if dev_dataset is None:
        dev_dataset = load_dev_dataset(config)

    val_ids = subject_ids or dev_dataset.get_subject_ids()
    if device is None:
        device = get_device()
    paths = create_output_dirs(config.output_dir)

    fold_results: list[FoldTrainingResult] = []
    per_subject: list[SubjectFoldMetrics] = []
    fold_mae_cm: list[float] = []
    pruned_config = False

    if enable_pruning and global_best_val_mae_cm is not None:
        print(
            f"Pruning ativo: MAE médio cm > {pruning_margin:.2f}× "
            f"melhor global ({global_best_val_mae_cm:.4f} cm) após {pruning_min_folds} folds"
        )

    print(f"LOSO — {len(val_ids)} folds | dispositivo: {device}")
    print(f"Sujeitos dev: {val_ids}")
    fold_timer = RunTimer("LOSO")
    fold_timer.set_total(len(val_ids))
    print(f"⏱ Cronômetro ativo — tempo/ETA após cada fold\n")
    if skip_existing:
        print("Modo retomada: folds com checkpoint existente serão pulados.\n")
    else:
        print()

    for i, val_sid in enumerate(val_ids, start=1):
        fold_timer.mark_step_start()
        print(f"[{i}/{len(val_ids)}] Fold val={val_sid}")

        loaders, fold = prepare_loso_fold(
            config, train_cfg, val_subject_id=val_sid, dev_dataset=dev_dataset
        )

        ckpt_path = _fold_checkpoint_path(paths["checkpoints"], val_sid, prefix=artifact_prefix)
        if skip_existing and ckpt_path.is_file():
            existing = load_existing_fold_metrics(
                paths["metrics"],
                val_sid,
                n_train=len(loaders.train_dataset),
                n_val=len(loaders.val_dataset),
                prefix=artifact_prefix,
            )
            if existing is not None:
                per_subject.append(existing)
                fold_timer.mark_step_end()
                fold_timer.step_done()
                print(
                    f"  → pulado (já treinado) | MAE={existing.mae:.4f} | "
                    f"RMSE={existing.rmse:.4f} | {fold_timer.status()}\n"
                )
                continue
            print("  → checkpoint existe, mas métricas ausentes — retreinando")

        result = train_single_fold(
            loaders=loaders,
            config=config,
            train_cfg=train_cfg,
            fold=fold,
            device=device,
            artifact_prefix=artifact_prefix,
            verbose=verbose,
            save_plot=save_individual_plots,
            enable_pruning=False,
        )

        sub_metrics = _metrics_from_result(
            result,
            n_train=len(loaders.train_dataset),
            n_val=len(loaders.val_dataset),
        )
        per_subject.append(sub_metrics)
        fold_results.append(result)

        if result.y_true_scaled is not None and result.y_pred_scaled is not None:
            yt_cm, yp_cm = metrics_to_original_scale(
                result.y_true_scaled, result.y_pred_scaled, fold.bundle
            )
            mae_cm = float(curve_window_mae(yt_cm, yp_cm))
        else:
            mae_cm = sub_metrics.mae
        fold_mae_cm.append(mae_cm)

        fold_timer.mark_step_end()
        fold_timer.step_done()
        print(
            f"  → MAE={sub_metrics.mae:.4f} (norm) | {mae_cm:.4f} cm | "
            f"RMSE={sub_metrics.rmse:.4f} | RMS={sub_metrics.rms:.4f} | "
            f"R²={sub_metrics.r2:.4f} | bias={sub_metrics.bias:+.4f} | "
            f"{fold_timer.status()}"
        )

        if (
            enable_pruning
            and global_best_val_mae_cm is not None
            and len(fold_mae_cm) >= pruning_min_folds
        ):
            running_mean_cm = float(np.mean(fold_mae_cm))
            threshold = global_best_val_mae_cm * pruning_margin
            if running_mean_cm > threshold:
                pruned_config = True
                print(
                    f"  → configuração interrompida por pruning após {len(fold_mae_cm)} folds "
                    f"(MAE médio={running_mean_cm:.4f} cm > {threshold:.4f} cm) | "
                    f"{fold_timer.status()}\n"
                )
                break
        print()

    if not per_subject:
        raise RuntimeError("Nenhum fold concluído — verifique dados e split.")

    per_subject.sort(key=lambda s: s.val_subject_id)
    summary = aggregate_loso_metrics(per_subject)
    _save_loso_artifacts(summary, fold_results, config, train_cfg, paths, artifact_prefix=artifact_prefix)
    _save_oof_predictions(
        fold_results, paths, config, train_cfg, dev_dataset, artifact_prefix=artifact_prefix
    )
    return fold_results, summary, pruned_config


def _save_oof_predictions(
    fold_results: list[FoldTrainingResult],
    paths: dict[str, Path],
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    dev_dataset: LoadedDataset,
    artifact_prefix: str = "etapa13",
) -> None:
    """Salva predições OOF em cm (formato longo para curvas)."""
    from analysis_utils import collect_oof_predictions_cm

    rows: list[dict] = []

    for result in fold_results:
        if result.y_true_scaled is None or result.y_pred_scaled is None:
            continue
        scaler_path = paths["scalers"] / f"{artifact_prefix}_{result.fold_name}_scaler.joblib"
        if not scaler_path.is_file():
            continue
        from etapa06_normalize import ScalerBundle

        bundle = ScalerBundle.load(scaler_path)
        yt_cm, yp_cm = metrics_to_original_scale(
            result.y_true_scaled, result.y_pred_scaled, bundle
        )
        times = result.window_start_times
        if times is None:
            times = np.arange(len(yt_cm), dtype=float) * config.stride_seconds
        rows.extend(
            predictions_to_long_rows(
                subject_id=result.val_subject_id,
                window_start_times=times,
                y_true_cm=yt_cm,
                y_pred_cm=yp_cm,
                sampling_hz=config.sampling_hz,
                window_samples=config.window_samples,
            )
        )

    if not rows:
        yt, yp, sids, _ = collect_oof_predictions_cm(
            config, train_cfg, dev_dataset, paths, artifact_prefix
        )
        if yt.size:
            for i in range(len(yt)):
                rows.append(
                    {
                        "subject_id": str(sids[i]),
                        "window_start_time": float(i),
                        "time_s": float(i),
                        "sample_in_window": 0,
                        "y_true_cm": float(yt[i]),
                        "y_pred_cm": float(yp[i]),
                        "error_cm": float(yp[i] - yt[i]),
                    }
                )

    if rows:
        df = pd.DataFrame(rows)
        out = paths["predictions"] / f"{artifact_prefix}_oof_predictions.csv"
        df.to_csv(out, index=False)


# ---------------------------------------------------------------------------
# 3. Persistência
# ---------------------------------------------------------------------------
def _save_loso_artifacts(
    summary: LOSOSummary,
    fold_results: list[FoldTrainingResult],
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    paths: dict[str, Path],
    artifact_prefix: str = "etapa13",
) -> None:
    per_subject_rows = [asdict(s) for s in summary.per_subject]
    df = pd.DataFrame(per_subject_rows)
    csv_path = paths["metrics"] / f"{artifact_prefix}_loso_per_subject.csv"
    df.to_csv(csv_path, index=False)

    payload = {
        "description": "LOSO completo nos sujeitos de desenvolvimento (70%). Teste final intocado.",
        "artifact_prefix": artifact_prefix,
        "model_type": config.model_type,
        "n_folds": summary.n_folds,
        "dev_subject_ids": summary.dev_subject_ids,
        "aggregate": {
            "mae_mean": summary.mean_mae,
            "mae_std": summary.std_mae,
            "rmse_mean": summary.mean_rmse,
            "rmse_std": summary.std_rmse,
            "r2_mean": summary.mean_r2,
            "r2_std": summary.std_r2,
            "bias_mean": summary.mean_bias,
            "bias_std": summary.std_bias,
        },
        "per_subject": per_subject_rows,
        "training_config": asdict(train_cfg),
        "checkpoints": [str(r.checkpoint_path) for r in fold_results],
    }
    json_path = paths["metrics"] / f"{artifact_prefix}_loso_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Relatório legível
# ---------------------------------------------------------------------------
def print_loso_report(summary: LOSOSummary, metrics_dir: Path) -> None:
    print("=" * 60)
    print("ETAPA 13 — LOSO completo nos 70% de desenvolvimento")
    print("=" * 60)
    print(f"Folds concluídos  : {summary.n_folds}")
    print(f"Métricas em escala vicon normalizada (amplitude por janela)\n")

    print(f"  {'ID':<6} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Bias':>8}")
    print("  " + "-" * 42)
    for s in summary.per_subject:
        print(f"  {s.val_subject_id:<6} {s.mae:8.4f} {s.rmse:8.4f} {s.r2:8.4f} {s.bias:+8.4f}")

    print("\nAgregado (média ± desvio-padrão entre sujeitos):")
    print(f"  MAE  : {summary.mean_mae:.4f} ± {summary.std_mae:.4f}")
    print(f"  RMSE : {summary.mean_rmse:.4f} ± {summary.std_rmse:.4f}")
    print(f"  R²   : {summary.mean_r2:.4f} ± {summary.std_r2:.4f}")
    print(f"  Bias : {summary.mean_bias:+.4f} ± {summary.std_bias:.4f}")

    print("\nArquivos salvos:")
    print(f"  {metrics_dir / 'etapa13_loso_per_subject.csv'}")
    print(f"  {metrics_dir / 'etapa13_loso_summary.json'}")
    print(f"  checkpoints/etapa13_loso_val_*_best.pt")

    print("\nInterpretação:")
    print("  - cada linha = desempenho em um sujeito nunca visto naquele fold")
    print("  - média ± std estima generalização dentro do grupo dev")
    print("  - use estes números para escolher hiperparâmetros (Etapa 14)")
    print("  - teste final (30%) permanece intocado")

    print("\nPróxima etapa: escolha de hiperparâmetros (Etapa 14).")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 5. Ponto de entrada da Etapa 13
# ---------------------------------------------------------------------------
def run_stage13_loso(
    config: ExperimentConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    verbose: bool = False,
    save_individual_plots: bool = False,
    skip_existing: bool = False,
    device: torch.device | None = None,
) -> tuple[list[FoldTrainingResult], LOSOSummary]:
    """
    Executa LOSO completo nos sujeitos de desenvolvimento (~19 folds).

    Pode levar ~20–40 min em CPU. Use skip_existing=True para retomar.
    """
    if config is None:
        config = build_default_config()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    paths = create_output_dirs(config.output_dir)
    dev_dataset = load_dev_dataset(config)

    print(f"Modelo: {config.model_type.upper()} | Loss: {train_cfg.loss.upper()}")
    print(f"Sujeitos dev: {len(dev_dataset.get_subject_ids())}")
    print("Teste final (30%): NÃO utilizado\n")

    fold_results, summary, _ = run_loso_on_dev(
        config=config,
        train_cfg=train_cfg,
        dev_dataset=dev_dataset,
        verbose=verbose,
        save_individual_plots=save_individual_plots,
        skip_existing=skip_existing,
        device=device,
    )

    print_loso_report(summary, paths["metrics"])
    return fold_results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 13 — LOSO completo nos 70%% de desenvolvimento (sem teste final)."
    )
    parser.add_argument("--verbose", action="store_true", help="log de épocas por fold")
    parser.add_argument("--plots", action="store_true", help="salvar curva de loss de cada fold")
    parser.add_argument(
        "--model",
        choices=("tcn", "cnn1d"),
        default="tcn",
        help="arquitetura (padrão: tcn)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="pular folds cujo checkpoint etapa13 já existe (retomar run)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_default_config()
    cfg.model_type = args.model
    run_stage13_loso(
        config=cfg,
        verbose=args.verbose,
        save_individual_plots=args.plots,
        skip_existing=args.skip_existing,
    )
