#!/usr/bin/env python3
"""
Etapa 15 — Treino final nos 70% + teste nos 30% intocados
=========================================================
Treina com TODOS os sujeitos de desenvolvimento (scaler ajustado só neles),
avalia UMA ÚNICA VEZ no grupo de teste final. Não reajusta nada após ver o teste.

NÃO executa automaticamente — rode quando estiver pronto.

Como rodar no futuro
--------------------
    cd /Users/Rodacki/Desktop/Hoffmann/UTT

    # conferir split e config (sem treinar)
    .venv/bin/python pipeline/etapa15_final_test.py --list

    # treino final + avaliação no teste (épocas automáticas via LOSO Etapa 13)
    .venv/bin/python pipeline/etapa15_final_test.py --run --epochs auto

    # épocas fixas
    .venv/bin/python pipeline/etapa15_final_test.py --run --epochs 20

    # usar hiperparâmetros da Etapa 14 (se existir best_hyperparameters.json)
    .venv/bin/python pipeline/etapa15_final_test.py --run --epochs auto --use-best-hparams

Pré-requisitos recomendados:
    - Etapa 5: split 70/30 salvo em outputs/pipeline_dl/splits/
    - Etapa 13 ou 14: LOSO dev para estimar número de épocas (--epochs auto)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs, get_device, set_seed
from etapa03_load import load_all_subjects
from etapa05_split import apply_split_to_dataset, load_subject_split
from etapa06_normalize import ScalerBundle, fit_scaler_bundle, transform_dataset
from etapa07_windows import WindowBatch, create_windows_from_dataset
from etapa08_dataloader import IMUWindowDataset, collate_imu_windows, create_dataloader
from etapa11_loss_metrics import (
    RegressionMetrics,
    TrainingConfig,
    build_criterion,
    build_optimizer,
    compute_regression_metrics,
    evaluate_loader,
    metrics_to_original_scale,
)
from etapa_evaluation import apply_calibration
from etapa12_train_fold import build_model_from_config, create_grad_scaler, train_one_epoch


# ---------------------------------------------------------------------------
# 1. Carregar configurações
# ---------------------------------------------------------------------------
def load_best_config(configs_dir: Path) -> dict[str, Any]:
    path = configs_dir / "best_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"best_config.json não encontrado em {configs_dir}. "
            "Rode a Etapa 14 (--with-hyperparams-fast / --best-quality) ou desative use_best_hparams."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_best_hyperparameters(configs_dir: Path) -> dict[str, Any] | None:
    for name in ("best_config.json", "best_hyperparameters.json"):
        path = configs_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def resolve_final_epochs(
    *,
    n_epochs_manual: int | None,
    use_best_hparams: bool,
    configs_dir: Path,
    metrics_dir: Path,
    hyperparams_from_search: bool = False,
) -> tuple[int, str]:
    """
    Define épocas do treino final.

    Prioridade: --final-epochs > best_config_p75 > fallback_loso (só sem busca).
    """
    if n_epochs_manual is not None:
        return int(n_epochs_manual), "manual"

    if use_best_hparams or hyperparams_from_search:
        best = load_best_config(configs_dir)
        p75 = best.get("best_epoch_p75") or best.get("final_epochs_p75")
        if p75 is not None:
            return int(p75), "best_config_p75"
        epochs = best.get("best_epoch_by_fold") or best.get("per_fold_best_epochs") or []
        if epochs:
            return int(np.percentile([int(e) for e in epochs], 75)), "best_config_p75"

    if hyperparams_from_search:
        raise FileNotFoundError(
            "best_config.json sem best_epoch_p75. Rode a busca progressiva completa."
        )

    return infer_epochs_from_loso(metrics_dir), "fallback_loso"


def apply_best_hyperparameters(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    payload: dict[str, Any],
) -> tuple[ExperimentConfig, TrainingConfig]:
    exp = payload.get("experiment", {})
    tr = payload.get("training", {})

    # Campos no topo de best_config.json
    if "model_type" in payload:
        config.model_type = payload["model_type"]
    if "window_seconds" in payload:
        config.window_seconds = float(payload["window_seconds"])
    if "stride_seconds" in payload:
        config.stride_seconds = float(payload["stride_seconds"])

    if "model_type" in exp:
        config.model_type = exp["model_type"]
    if "window_seconds" in exp:
        config.window_seconds = float(exp["window_seconds"])
    if "stride_seconds" in exp:
        config.stride_seconds = float(exp["stride_seconds"])
    if "target_mode" in payload:
        config.target_mode = payload["target_mode"]
    if "target_mode" in exp:
        config.target_mode = exp["target_mode"]

    merged_tr = {**tr}
    for key in (
        "loss",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "huber_delta",
        "dropout",
        "min_delta",
    ):
        if key in payload:
            merged_tr[key] = payload[key]

    for key in ("loss", "learning_rate", "weight_decay", "batch_size", "max_epochs", "patience", "huber_delta", "dropout", "min_delta"):
        if key in merged_tr:
            setattr(train_cfg, key, merged_tr[key])

    return config, train_cfg


def infer_epochs_from_loso(metrics_dir: Path, percentile: float = 75.0, prefix: str = "etapa13") -> int:
    """
    Usa percentil de best_epoch do LOSO dev como duração do treino final.

    Padrão: percentil 75 para evitar subtreino.
    """
    per_subject = metrics_dir / f"{prefix}_loso_per_subject.csv"
    summary = metrics_dir / f"{prefix}_loso_summary.json"

    epochs: list[int] = []
    if per_subject.is_file():
        df = pd.read_csv(per_subject)
        if "best_epoch" in df.columns and len(df):
            epochs = df["best_epoch"].astype(int).tolist()

    if not epochs and summary.is_file():
        payload = json.loads(summary.read_text(encoding="utf-8"))
        epochs = [int(s["best_epoch"]) for s in payload.get("per_subject", []) if "best_epoch" in s]

    # Tentar best_config.json para épocas da melhor config
    best_cfg_path = metrics_dir.parent / "configs" / "best_config.json"
    if best_cfg_path.is_file():
        best_cfg = json.loads(best_cfg_path.read_text(encoding="utf-8"))
        if best_cfg.get("final_epochs_p75"):
            return int(best_cfg["final_epochs_p75"])
        if best_cfg.get("per_fold_best_epochs"):
            epochs = [int(e) for e in best_cfg["per_fold_best_epochs"]]

    if epochs:
        return int(np.percentile(epochs, percentile))

    return 20  # fallback conservador


# ---------------------------------------------------------------------------
# 2. Preparar dados dev/teste
# ---------------------------------------------------------------------------
def prepare_final_datasets(
    config: ExperimentConfig,
    normalize_target: bool = True,
) -> tuple[WindowBatch, WindowBatch, ScalerBundle, list[str], list[str]]:
    """
    Ajusta scaler nos 70% dev, transforma dev e teste, cria janelas.

    Retorna: (train_windows, test_windows, scaler, dev_ids, test_ids)
    """
    paths = create_output_dirs(config.output_dir)
    dataset = load_all_subjects(config.data_dir)
    split = load_subject_split(paths["splits"])
    dev_dataset, test_dataset = apply_split_to_dataset(dataset, split)

    dev_ids = dev_dataset.get_subject_ids()
    test_ids = test_dataset.get_subject_ids()

    overlap = set(dev_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Vazamento: sujeitos em dev e teste: {sorted(overlap)}")

    bundle = fit_scaler_bundle(
        dev_dataset,
        train_subject_ids=dev_ids,
        feature_cols=config.feature_cols,
        normalize_target=normalize_target,
    )
    bundle.fitted_on_subject_ids = dev_ids

    dev_norm = transform_dataset(dev_dataset, bundle, subject_ids=dev_ids, normalize_target=normalize_target)
    test_norm = transform_dataset(
        test_dataset, bundle, subject_ids=test_ids, normalize_target=normalize_target
    )

    train_batch = create_windows_from_dataset(
        dev_norm,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=config.target_mode,
    )
    test_batch = create_windows_from_dataset(
        test_norm,
        window_samples=config.window_samples,
        stride_samples=config.stride_samples,
        feature_cols=config.feature_cols,
        target_mode=config.target_mode,
    )

    return train_batch, test_batch, bundle, dev_ids, test_ids


# ---------------------------------------------------------------------------
# 3. Treino final (100% dev — sem validação interna)
# ---------------------------------------------------------------------------
def train_final_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int,
    verbose: bool = True,
    *,
    use_amp: bool = False,
) -> list[float]:
    """Treina em todas as janelas de desenvolvimento por n_epochs fixas."""
    grad_scaler, amp_ok = create_grad_scaler(device, use_amp)
    if use_amp and not amp_ok:
        print("  Aviso: --amp solicitado mas indisponível no treino final; usando FP32.")
    losses: list[float] = []
    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp=use_amp,
            grad_scaler=grad_scaler,
            amp_enabled=amp_ok,
        )
        losses.append(loss)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
            print(f"  época {epoch:3d}/{n_epochs} | train_loss={loss:.4f}")
    return losses


# ---------------------------------------------------------------------------
# 4. Predições detalhadas no teste
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_test_predictions(
    model: nn.Module,
    test_batch: WindowBatch,
    loader: DataLoader,
    device: torch.device,
    scaler: ScalerBundle,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Coleta predições com tempo absoluto (escalar ou curva por janela)."""
    from curve_utils import predictions_to_long_rows

    model.eval()
    rows: list[dict] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        y_hat = model(x)

        y_np = y.detach().cpu().numpy()
        yhat_np = y_hat.detach().cpu().numpy()
        n = y_np.shape[0]

        sids = batch["subject_ids"]
        times = batch["window_start_time"].cpu().numpy()

        y_cm, yhat_cm = metrics_to_original_scale(y_np, yhat_np, scaler)

        for i in range(n):
            sid = sids[i]
            t_win = np.array([float(times[i])])
            if y_cm.ndim > 1:
                yt_i = y_cm[i : i + 1]
                yp_i = yhat_cm[i : i + 1]
            else:
                yt_i = np.asarray([y_cm[i]])
                yp_i = np.asarray([yhat_cm[i]])
            rows.extend(
                predictions_to_long_rows(
                    subject_id=sid,
                    window_start_times=t_win,
                    y_true_cm=yt_i,
                    y_pred_cm=yp_i,
                    sampling_hz=config.sampling_hz,
                    window_samples=config.window_samples,
                )
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Relatórios
# ---------------------------------------------------------------------------
def print_list_report(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    dev_ids: list[str],
    test_ids: list[str],
    train_batch: WindowBatch,
    test_batch: WindowBatch,
    n_epochs_auto: int | None,
    best_hparams: dict | None,
) -> None:
    print("=" * 60)
    print("ETAPA 15 — Treino final + teste intocado (prévia)")
    print("=" * 60)
    print(f"Modelo           : {config.model_type.upper()}")
    print(f"Janela / stride  : {config.window_seconds}s / {config.stride_seconds}s")
    print(f"Loss / lr        : {train_cfg.loss} / {train_cfg.learning_rate}")
    print(f"Dev  ({len(dev_ids)}): {dev_ids}")
    print(f"Test ({len(test_ids)}): {test_ids}")
    print(f"Janelas treino   : {train_batch.n_windows}")
    print(f"Janelas teste    : {test_batch.n_windows}")
    if best_hparams:
        print(f"Hiperparâmetros  : best_hyperparameters.json (trial {best_hparams.get('best_trial_id')})")
    else:
        print("Hiperparâmetros  : padrão (best_hyperparameters.json não encontrado)")
    if n_epochs_auto is not None:
        print(f"Épocas (auto)    : {n_epochs_auto} (mediana LOSO Etapa 13)")
    print("\nPara executar: .venv/bin/python pipeline/etapa15_final_test.py --run --epochs auto")
    print("=" * 60)


def print_final_test_report(
    metrics_scaled: RegressionMetrics,
    metrics_cm: RegressionMetrics,
    test_ids: list[str],
    n_epochs: int,
    paths: dict[str, Path],
) -> None:
    print("\n" + "=" * 60)
    print("ETAPA 15 — Resultado no teste final (30%)")
    print("=" * 60)
    print(f"Épocas de treino  : {n_epochs} (100% janelas dev)")
    print(f"Sujeitos teste    : {test_ids}")
    print("\nMétricas (escala normalizada do alvo):")
    print(f"  MAE  : {metrics_scaled.mae:.4f}")
    print(f"  RMSE : {metrics_scaled.rmse:.4f}")
    print(f"  R²   : {metrics_scaled.r2:.4f}")
    print(f"  Bias : {metrics_scaled.bias:+.4f}")
    print(f"  MAPE : {metrics_scaled.mape_pct:.2f}%")
    print("\nMétricas (cm — interpretação clínica):")
    print(f"  MAE  : {metrics_cm.mae:.4f} cm")
    print(f"  RMSE : {metrics_cm.rmse:.4f} cm")
    print(f"  RMS  : {metrics_cm.rms:.4f} cm")
    print(f"  R²   : {metrics_cm.r2:.4f}")
    print(f"  Bias : {metrics_cm.bias:+.4f} cm")
    print(f"  MAPE : {metrics_cm.mape_pct:.2f}%")

    print("\nArquivos salvos:")
    print(f"  {paths['checkpoints'] / 'etapa15_final_model.pt'}")
    print(f"  {paths['scalers'] / 'etapa15_final_scaler.joblib'}")
    print(f"  {paths['predictions'] / 'etapa15_test_predictions.csv'}")
    print(f"  {paths['metrics'] / 'etapa15_test_metrics.json'}")
    print(f"  {paths['configs'] / 'etapa15_final_run.json'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. Execução principal
# ---------------------------------------------------------------------------
def run_stage15_final_test(
    config: ExperimentConfig | None = None,
    train_cfg: TrainingConfig | None = None,
    n_epochs: int | None = None,
    use_best_hparams: bool = False,
    verbose: bool = True,
    calibrate_from_loso: bool = False,
    device: torch.device | None = None,
    num_workers: int = 0,
    hyperparams_from_search: bool = False,
    epochs_origin: str | None = None,
) -> dict[str, Any]:
    """
    Treina modelo final nos 70% dev e avalia uma vez nos 30% teste.
    """
    if config is None:
        config = build_default_config()
    if train_cfg is None:
        train_cfg = TrainingConfig()

    paths = create_output_dirs(config.output_dir)
    if device is None:
        device = get_device()
    train_cfg.num_workers = num_workers

    best_hparams: dict[str, Any] | None = None
    if use_best_hparams:
        best_hparams = load_best_config(paths["configs"])
        config, train_cfg = apply_best_hyperparameters(config, train_cfg, best_hparams)

    if n_epochs is None:
        n_epochs, resolved_origin = resolve_final_epochs(
            n_epochs_manual=None,
            use_best_hparams=use_best_hparams,
            configs_dir=paths["configs"],
            metrics_dir=paths["metrics"],
            hyperparams_from_search=hyperparams_from_search,
        )
        epochs_origin = epochs_origin or resolved_origin
    else:
        epochs_origin = epochs_origin or "manual"

    set_seed(config.seed)

    print(f"Dispositivo: {device}")
    print(f"Épocas finais: {n_epochs} (origem: {epochs_origin})")
    print("Preparando dados (scaler fit nos dev, transform no teste)...")
    train_batch, test_batch, scaler, dev_ids, test_ids = prepare_final_datasets(config)

    train_ds = IMUWindowDataset.from_window_batch(train_batch)
    test_ds = IMUWindowDataset.from_window_batch(test_batch)
    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=getattr(train_cfg, "pin_memory", False),
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=getattr(train_cfg, "pin_memory", False),
    )

    print(f"Treino final: {len(train_ds)} janelas | Teste: {len(test_ds)} janelas\n")

    model = build_model_from_config(config, train_cfg).to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)

    print("Treinando em 100% do desenvolvimento...")
    train_losses = train_final_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        n_epochs=n_epochs,
        verbose=verbose,
        use_amp=getattr(train_cfg, "use_amp", False),
    )

    print("\nAvaliando teste final (única vez)...")
    test_loss, metrics_scaled, y_true, y_pred = evaluate_loader(
        model, test_loader, criterion, device
    )
    y_true_cm, y_pred_cm = metrics_to_original_scale(y_true, y_pred, scaler)
    metrics_cm = compute_regression_metrics(y_true_cm, y_pred_cm)

    pred_df = collect_test_predictions(model, test_batch, test_loader, device, scaler, config)

    # --- salvar artefatos ---
    ckpt_path = paths["checkpoints"] / "etapa15_final_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": config.model_type,
            "dev_subject_ids": dev_ids,
            "test_subject_ids": test_ids,
            "n_epochs": n_epochs,
            "train_loss_final": train_losses[-1] if train_losses else None,
            "test_metrics_scaled": metrics_scaled.to_dict(),
            "test_metrics_cm": metrics_cm.to_dict(),
        },
        ckpt_path,
    )
    scaler_path = paths["scalers"] / "etapa15_final_scaler.joblib"
    scaler.save(scaler_path)

    pred_path = paths["predictions"] / "etapa15_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    from etapa_evaluation import apply_test_calibration, save_window_and_subject_metrics

    eval_paths = save_window_and_subject_metrics(
        pred_df,
        paths["metrics"],
        paths["plots"],
        prefix="etapa15_test",
        skip_bland_altman=config.target_mode == "curve",
        plot_curve_overlay_flag=config.target_mode == "curve",
    )

    cal_metrics = None
    if calibrate_from_loso:
        from etapa_evaluation import run_calibration_from_loso_oof

        a, b, oof_path = run_calibration_from_loso_oof(config.output_dir)
        if oof_path is not None:
            cal_metrics = apply_test_calibration(pred_path, a, b, paths["metrics"])
            metrics_cm = compute_regression_metrics(
                pred_df["y_true_cm"],
                apply_calibration(pred_df["y_pred_cm"].to_numpy(), a, b),
            )

    metrics_payload = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Avaliação única no teste final. Não reajustar hiperparâmetros após isto.",
        "n_epochs": n_epochs,
        "epochs_origin": epochs_origin,
        "dev_subject_ids": dev_ids,
        "test_subject_ids": test_ids,
        "test_loss": test_loss,
        "metrics_scaled": metrics_scaled.to_dict(),
        "metrics_cm": metrics_cm.to_dict(),
        "experiment": {
            "model_type": config.model_type,
            "window_seconds": config.window_seconds,
            "stride_seconds": config.stride_seconds,
            "window_samples": config.window_samples,
            "stride_samples": config.stride_samples,
        },
        "training": asdict(train_cfg),
        "used_best_hyperparameters": best_hparams is not None,
        "best_trial_id": best_hparams.get("best_trial_id") or best_hparams.get("config_id") if best_hparams else None,
        "calibration": cal_metrics,
        "evaluation_files": {k: str(v) for k, v in eval_paths.items()},
    }
    metrics_path = paths["metrics"] / "etapa15_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    run_config_path = paths["configs"] / "etapa15_final_run.json"
    run_config_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print_final_test_report(metrics_scaled, metrics_cm, test_ids, n_epochs, paths)

    return metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 15 — treino final (70% dev) + teste intocado (30%)."
    )
    parser.add_argument("--list", action="store_true", help="mostrar split/config e sair")
    parser.add_argument("--run", action="store_true", help="executar treino final + teste")
    parser.add_argument(
        "--epochs",
        type=str,
        default="auto",
        help="'auto' (mediana LOSO Etapa 13) ou número inteiro",
    )
    parser.add_argument(
        "--use-best-hparams",
        action="store_true",
        help="carregar outputs/.../configs/best_hyperparameters.json (Etapa 14)",
    )
    parser.add_argument("--verbose", action="store_true", help="log de épocas de treino")
    return parser.parse_args()


def _parse_epochs(value: str, metrics_dir: Path) -> int:
    if value.lower() == "auto":
        return infer_epochs_from_loso(metrics_dir)
    return int(value)


if __name__ == "__main__":
    args = parse_args()
    cfg = build_default_config()
    train = TrainingConfig()
    paths = create_output_dirs(cfg.output_dir)

    best = load_best_hyperparameters(paths["configs"]) if args.use_best_hparams else None
    if best:
        cfg, train = apply_best_hyperparameters(cfg, train, best)

    n_epochs_auto = infer_epochs_from_loso(paths["metrics"]) if args.epochs.lower() == "auto" else None

    if args.list or not args.run:
        train_batch, test_batch, _, dev_ids, test_ids = prepare_final_datasets(cfg)
        print_list_report(cfg, train, dev_ids, test_ids, train_batch, test_batch, n_epochs_auto, best)
    else:
        n_epochs = _parse_epochs(args.epochs, paths["metrics"])
        run_stage15_final_test(
            config=cfg,
            train_cfg=train,
            n_epochs=n_epochs,
            use_best_hparams=args.use_best_hparams,
            verbose=args.verbose,
        )
