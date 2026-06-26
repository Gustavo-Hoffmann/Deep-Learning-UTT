#!/usr/bin/env python3
"""
Etapa 14 — Busca progressiva de hiperparâmetros (coarse → refine → stability).
Substitui o grid bruto por busca inteligente com ranking composto.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis_utils import (
    collect_oof_predictions_cm,
    composite_score,
    metrics_from_cm_arrays,
    proportional_bias_analysis,
)
from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa11_loss_metrics import LossName, TrainingConfig, metrics_to_original_scale
from etapa13_loso import load_dev_dataset, run_loso_on_dev
from etapa14_hyperparameters import HyperparamTrial, apply_trial, configs_to_dict
from run_session import RunSession
from timing_utils import RunTimer, format_duration

ModelType = Literal["tcn", "cnn1d"]
STABILITY_SEEDS = (7, 42, 123, 2026, 3407)


@dataclass
class SearchConfig:
    """Uma configuração candidata na busca progressiva."""

    config_id: str
    model_type: ModelType
    window_seconds: float
    stride_seconds: float
    learning_rate: float
    dropout: float
    loss: LossName
    batch_size: int
    seed: int = 42
    phase: str = "fast"
    description: str = ""

    def to_trial(self) -> HyperparamTrial:
        return HyperparamTrial(
            trial_id=self.config_id,
            description=self.description or self.config_id,
            model_type=self.model_type,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
            loss=self.loss,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
        )

    def fingerprint(self) -> str:
        parts = (
            self.model_type,
            f"{self.window_seconds:.2f}",
            f"{self.stride_seconds:.2f}",
            f"{self.learning_rate:.1e}",
            f"{self.dropout:.2f}",
            self.loss,
            str(self.batch_size),
            str(self.seed),
        )
        return "|".join(parts)


@dataclass
class ConfigResult:
    config: SearchConfig
    n_folds: int
    mae_mean_cm: float
    mae_std_cm: float
    rmse_mean_cm: float
    rmse_std_cm: float
    bias_cm: float
    proportional_bias_slope: float
    composite: float
    rms_mean_cm: float | None = None
    r2_mean: float | None = None
    status: str = "completed"
    stop_reason: str | None = None
    seed_mae_mean: float | None = None
    seed_mae_std: float | None = None
    per_fold_best_epochs: list[int] = field(default_factory=list)
    experiment: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "config_id": self.config.config_id,
            "phase": self.config.phase,
            "model_type": self.config.model_type,
            "window_seconds": self.config.window_seconds,
            "stride_seconds": self.config.stride_seconds,
            "learning_rate": self.config.learning_rate,
            "dropout": self.config.dropout,
            "loss": self.config.loss,
            "batch_size": self.config.batch_size,
            "seed": self.config.seed,
            "n_folds": self.n_folds,
            "mae_mean_cm": self.mae_mean_cm,
            "mae_std_cm": self.mae_std_cm,
            "rmse_mean_cm": self.rmse_mean_cm,
            "rmse_std_cm": self.rmse_std_cm,
            "rms_mean_cm": self.rms_mean_cm,
            "bias_cm": self.bias_cm,
            "proportional_bias_slope": self.proportional_bias_slope,
            "composite_score": self.composite,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "seed_mae_mean": self.seed_mae_mean,
            "seed_mae_std": self.seed_mae_std,
        }


def _make_config_id(prefix: str, cfg: SearchConfig) -> str:
    h = hashlib.md5(cfg.fingerprint().encode()).hexdigest()[:8]
    return f"{prefix}_{cfg.model_type}_{h}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _result_from_csv_row(row: dict[str, Any]) -> ConfigResult:
    cfg = SearchConfig(
        config_id=str(row["config_id"]),
        model_type=row["model_type"],  # type: ignore[arg-type]
        window_seconds=float(row["window_seconds"]),
        stride_seconds=float(row["stride_seconds"]),
        learning_rate=float(row["learning_rate"]),
        dropout=float(row["dropout"]),
        loss=row["loss"],  # type: ignore[arg-type]
        batch_size=int(row["batch_size"]),
        seed=int(row.get("seed", 42)),
        phase=str(row.get("phase", "fast")),
        description=str(row.get("description", row["config_id"])),
    )
    return ConfigResult(
        config=cfg,
        n_folds=int(row.get("n_folds", 0)),
        mae_mean_cm=float(row["mae_mean_cm"]),
        mae_std_cm=float(row.get("mae_std_cm", 0.0)),
        rmse_mean_cm=float(row["rmse_mean_cm"]),
        rmse_std_cm=float(row.get("rmse_std_cm", 0.0)),
        rms_mean_cm=_optional_float(row.get("rms_mean_cm")),
        bias_cm=float(row.get("bias_cm", 0.0)),
        proportional_bias_slope=float(row.get("proportional_bias_slope", 0.0)),
        composite=float(row.get("composite_score", row.get("composite", 0.0))),
        r2_mean=_optional_float(row.get("r2_mean")),
        status=str(row.get("status", "completed")),
        stop_reason=row.get("stop_reason") if pd.notna(row.get("stop_reason")) else None,
        seed_mae_mean=_optional_float(row.get("seed_mae_mean")),
        seed_mae_std=_optional_float(row.get("seed_mae_std")),
    )


def load_phase_results_from_csv(
    run_dir: Path,
    filename: str,
    phase: str,
) -> list[ConfigResult]:
    """Carrega resultados de uma fase já gravados em CSV (necessário no --resume)."""
    path = run_dir / filename
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    results: list[ConfigResult] = []
    for _, row in df.iterrows():
        if str(row.get("phase", phase)) != phase:
            continue
        results.append(_result_from_csv_row(row.to_dict()))
    return results


def _merge_phase_results(
    in_memory: list[ConfigResult],
    from_csv: list[ConfigResult],
) -> list[ConfigResult]:
    """Une resultados da sessão atual com os do CSV, sem duplicar config_id."""
    by_id = {r.config.config_id: r for r in from_csv}
    for r in in_memory:
        by_id[r.config.config_id] = r
    return list(by_id.values())


def generate_smoke_grid() -> list[SearchConfig]:
    """Grade mínima para --smoke-test (validação rápida do fluxo)."""
    configs: list[SearchConfig] = []
    for model in ("tcn", "cnn1d"):
        cfg = SearchConfig(
            config_id="",
            model_type=model,  # type: ignore[arg-type]
            window_seconds=2.0,
            stride_seconds=1.0,
            learning_rate=1e-3,
            dropout=0.2,
            loss="huber",
            batch_size=32,
            phase="fast",
            description=f"smoke {model} win=2s",
        )
        cfg.config_id = _make_config_id("smoke", cfg)
        configs.append(cfg)
    return configs


def build_best_config_payload(
    best: ConfigResult,
    *,
    backend: str,
    stability_seeds: bool,
    seeds_used: list[int] | None = None,
    base_train: TrainingConfig | None = None,
) -> dict[str, Any]:
    """Monta best_config.json central com todos os campos exigidos."""
    epochs = [int(e) for e in best.per_fold_best_epochs if e]
    tr = best.training or {}
    exp = best.experiment or {}
    r2 = getattr(best, "r2_mean", None)

    payload: dict[str, Any] = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "selection_criterion": "composite_score (MAE + std + bias + slope + RMSE/RMS por janela)",
        "target_mode": exp.get("target_mode", "curve"),
        "config_id": best.config.config_id,
        "source": best.config.phase,
        "model_type": best.config.model_type,
        "window_seconds": best.config.window_seconds,
        "stride_seconds": best.config.stride_seconds,
        "learning_rate": best.config.learning_rate,
        "dropout": best.config.dropout,
        "loss": best.config.loss,
        "batch_size": best.config.batch_size,
        "weight_decay": tr.get("weight_decay", base_train.weight_decay if base_train else 1e-4),
        "max_epochs": tr.get("max_epochs", base_train.max_epochs if base_train else 300),
        "patience": tr.get("patience", base_train.patience if base_train else 30),
        "min_delta": tr.get("min_delta", base_train.min_delta if base_train else 1e-4),
        "best_epoch_by_fold": epochs,
        "best_epoch_mean": float(np.mean(epochs)) if epochs else None,
        "best_epoch_median": float(np.median(epochs)) if epochs else None,
        "best_epoch_p75": int(np.percentile(epochs, 75)) if epochs else None,
        "final_epochs_p75": int(np.percentile(epochs, 75)) if epochs else None,
        "mae_mean_cm": best.mae_mean_cm,
        "mae_std_cm": best.mae_std_cm,
        "rmse_mean_cm": best.rmse_mean_cm,
        "rmse_std_cm": best.rmse_std_cm,
        "rms_mean_cm": best.rms_mean_cm,
        "r2_mean": r2,
        "bias_cm": best.bias_cm,
        "proportional_bias_slope": best.proportional_bias_slope,
        "composite_score": best.composite,
        "seeds_used": seeds_used if stability_seeds and seeds_used else None,
        "backend": backend,
        "experiment": exp,
        "training": tr,
        "per_fold_best_epochs": epochs,
    }
    return payload


def generate_fast_grid() -> list[SearchConfig]:
    """Busca rápida/coarse conforme especificação."""
    configs: list[SearchConfig] = []
    for model, win, stride, lr, drop in itertools.product(
        ("cnn1d", "tcn"),
        (1.0, 2.0, 3.0, 4.0),
        (0.5, 1.0),
        (1e-3, 5e-4),
        (0.1, 0.2),
    ):
        cfg = SearchConfig(
            config_id="",
            model_type=model,  # type: ignore[arg-type]
            window_seconds=win,
            stride_seconds=stride,
            learning_rate=lr,
            dropout=drop,
            loss="huber",
            batch_size=32,
            phase="fast",
            description=f"{model} win={win}s stride={stride}s lr={lr} drop={drop}",
        )
        cfg.config_id = _make_config_id("fast", cfg)
        configs.append(cfg)
    return configs


def generate_refine_grid(top_configs: list[SearchConfig], tested: set[str]) -> list[SearchConfig]:
    """Variações ao redor dos melhores candidatos."""
    configs: list[SearchConfig] = []
    for base in top_configs:
        wins = sorted(
            {
                max(0.75, min(4.5, base.window_seconds + d))
                for d in (-0.5, 0.0, 0.5)
            }
        )
        for win, stride, lr, drop, loss, bs in itertools.product(
            wins,
            (0.25, 0.5, 1.0),
            (1e-3, 5e-4, 1e-4),
            (0.1, 0.2, 0.3),
            ("mse", "huber"),
            (16, 32, 64),
        ):
            cfg = SearchConfig(
                config_id="",
                model_type=base.model_type,
                window_seconds=win,
                stride_seconds=stride,
                learning_rate=lr,
                dropout=drop,
                loss=loss,  # type: ignore[arg-type]
                batch_size=bs,
                phase="refine",
                description=f"refine {base.model_type} win={win}s",
            )
            fp = cfg.fingerprint()
            if fp in tested:
                continue
            cfg.config_id = _make_config_id("ref", cfg)
            configs.append(cfg)
            tested.add(fp)
    return configs


def generate_stability_configs(top_configs: list[SearchConfig]) -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for base in top_configs:
        for seed in STABILITY_SEEDS:
            cfg = deepcopy(base)
            cfg.seed = seed
            cfg.phase = "stability"
            cfg.config_id = _make_config_id("stab", cfg)
            configs.append(cfg)
    return configs


def _collect_oof_cm(
    config: ExperimentConfig,
    train_cfg: TrainingConfig,
    dev_dataset,
    paths: dict[str, Path],
    artifact_prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    paths_dict = paths if isinstance(paths, dict) else create_output_dirs(config.output_dir)
    yt, yp, _, best_epochs = collect_oof_predictions_cm(
        config, train_cfg, dev_dataset, paths_dict, artifact_prefix
    )
    return yt, yp, best_epochs


def evaluate_search_config(
    search_cfg: SearchConfig,
    base_config: ExperimentConfig,
    base_train: TrainingConfig,
    dev_dataset,
    *,
    verbose: bool = False,
    skip_existing: bool = False,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    pruning_min_folds: int = 5,
    global_best_mae_cm: float | None = None,
    device=None,
    num_workers: int = 0,
) -> ConfigResult:
    """Avalia uma configuração via LOSO completo nos 70% dev."""
    trial = search_cfg.to_trial()
    config, train_cfg = apply_trial(base_config, base_train, trial)
    config.seed = search_cfg.seed
    train_cfg.dropout = search_cfg.dropout
    train_cfg.num_workers = num_workers

    prefix = f"etapa14_{search_cfg.config_id}"
    paths = create_output_dirs(config.output_dir)

    _, summary, pruned = run_loso_on_dev(
        config=config,
        train_cfg=train_cfg,
        dev_dataset=dev_dataset,
        verbose=verbose,
        save_individual_plots=True,
        skip_existing=skip_existing,
        artifact_prefix=prefix,
        device=device,
        enable_pruning=enable_pruning,
        pruning_warmup_epochs=pruning_warmup_epochs,
        pruning_margin=pruning_margin,
        pruning_min_folds=pruning_min_folds,
        global_best_val_mae_cm=global_best_mae_cm,
    )

    y_true_cm, y_pred_cm, best_epochs = _collect_oof_cm(
        config, train_cfg, dev_dataset, paths, prefix
    )
    if len(y_true_cm) == 0:
        mae_mean = summary.mean_mae
        rmse_mean = summary.mean_rmse
        bias = summary.mean_bias
        slope = 0.0
        mae_std = summary.std_mae
        rmse_std = summary.std_rmse
        r2_mean = None
        rms_mean = None
    else:
        m = metrics_from_cm_arrays(y_true_cm, y_pred_cm)
        mae_mean = m.mae
        rmse_mean = m.rmse
        rms_mean = m.rms
        bias = m.bias
        r2_mean = m.r2
        pb = proportional_bias_analysis(y_true_cm, y_pred_cm)
        slope = pb.slope
        mae_std = summary.std_mae
        rmse_std = summary.std_rmse

    comp = composite_score(mae_mean, mae_std, rmse_mean, bias, slope)
    exp, tr = configs_to_dict(config, train_cfg)
    tr["dropout"] = search_cfg.dropout

    return ConfigResult(
        config=search_cfg,
        n_folds=summary.n_folds,
        mae_mean_cm=mae_mean,
        mae_std_cm=mae_std,
        rmse_mean_cm=rmse_mean,
        rmse_std_cm=rmse_std,
        rms_mean_cm=rms_mean,
        bias_cm=bias,
        proportional_bias_slope=slope,
        composite=comp,
        r2_mean=r2_mean,
        status="pruned_early" if pruned else "completed",
        stop_reason="pruning" if pruned else None,
        per_fold_best_epochs=best_epochs,
        experiment=exp,
        training=tr,
    )


def run_progressive_search(
    base_config: ExperimentConfig | None = None,
    base_train: TrainingConfig | None = None,
    *,
    refine_top_k: int = 0,
    skip_refine: bool = False,
    stability_seeds: bool = False,
    resume: bool = False,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    pruning_min_folds: int = 5,
    verbose: bool = False,
    skip_existing: bool = False,
    device=None,
    num_workers: int = 0,
    smoke_test: bool = False,
    backend: str = "cpu",
) -> tuple[list[ConfigResult], ConfigResult, RunSession]:
    """Executa busca progressiva: fast → refine → stability."""
    if base_config is None:
        base_config = build_default_config()
    if base_train is None:
        base_train = TrainingConfig()

    paths = create_output_dirs(base_config.output_dir)
    session = RunSession(base_config.output_dir, resume=resume)
    session.open_log()
    session.save_config(
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "refine_top_k": refine_top_k,
            "skip_refine": skip_refine,
            "stability_seeds": stability_seeds,
            "pruning": enable_pruning,
            "smoke_test": smoke_test,
            "backend": backend,
            "training": asdict(base_train),
        }
    )

    dev_dataset = load_dev_dataset(base_config)
    all_results: list[ConfigResult] = []
    tested_fps: set[str] = set()
    global_best_mae: float | None = None
    seeds_used: list[int] = []
    search_timer = RunTimer("busca total")
    session.log("⏱ Cronômetro ativo — tempo e ETA após cada configuração")

    # --- Fase A: busca rápida ---
    fast_configs = generate_smoke_grid() if smoke_test else generate_fast_grid()
    session.log(f"Fase fast: {len(fast_configs)} configurações")
    done_fast = session.load_completed_config_ids("hyperparams_fast_results.csv") if resume else set()
    fast_timer = RunTimer("fast")
    fast_pending = sum(1 for c in fast_configs if c.config_id not in done_fast)
    fast_timer.set_total(fast_pending)

    for i, cfg in enumerate(fast_configs, 1):
        if cfg.config_id in done_fast:
            session.log(f"[fast {i}/{len(fast_configs)}] {cfg.config_id} — já concluído, pulando")
            continue
        fast_timer.mark_step_start()
        session.log(f"[fast {i}/{len(fast_configs)}] {cfg.config_id}")
        result = evaluate_search_config(
            cfg,
            base_config,
            base_train,
            dev_dataset,
            verbose=verbose,
            skip_existing=skip_existing or resume,
            enable_pruning=enable_pruning,
            pruning_warmup_epochs=pruning_warmup_epochs,
            pruning_margin=pruning_margin,
            pruning_min_folds=pruning_min_folds,
            global_best_mae_cm=global_best_mae,
            device=device,
            num_workers=num_workers,
        )
        all_results.append(result)
        session.append_results_csv("hyperparams_fast_results.csv", result.to_row())
        if result.status == "completed":
            global_best_mae = result.mae_mean_cm if global_best_mae is None else min(global_best_mae, result.mae_mean_cm)
        fast_timer.mark_step_end()
        fast_timer.step_done()
        session.log(
            f"  → MAE={result.mae_mean_cm:.4f} cm | score={result.composite:.4f} | "
            f"{fast_timer.status()} | {search_timer.status()}"
        )

    fast_results = _merge_phase_results(
        [r for r in all_results if r.config.phase == "fast"],
        load_phase_results_from_csv(session.run_dir, "hyperparams_fast_results.csv", "fast"),
    )
    fast_sorted = sorted(fast_results, key=lambda r: r.composite)
    if fast_results:
        global_best_mae = min(r.mae_mean_cm for r in fast_results if r.status == "completed")

    # --- Fase B: refinamento ---
    refine_results: list[ConfigResult] = []
    if skip_refine:
        refine_results = load_phase_results_from_csv(
            session.run_dir, "hyperparams_refined_results.csv", "refine"
        )
        session.log(
            "Fase refine: pulada (--skip-refine). "
            f"{len(refine_results)} resultado(s) parcial(is) mantidos no ranking."
        )
    elif refine_top_k > 0 and fast_sorted:
        top = [r.config for r in fast_sorted[:refine_top_k]]
        for r in fast_results:
            tested_fps.add(r.config.fingerprint())
        refine_configs = generate_refine_grid(top, tested_fps)
        session.log(f"Fase refine: {len(refine_configs)} configurações (top-{refine_top_k})")
        done_ref = session.load_completed_config_ids("hyperparams_refined_results.csv") if resume else set()
        if done_ref:
            session.log(f"  → {len(done_ref)} refine(s) já concluído(s); retomando pendentes")
        refine_timer = RunTimer("refine")
        refine_pending = sum(1 for c in refine_configs if c.config_id not in done_ref)
        refine_timer.set_total(refine_pending)

        for i, cfg in enumerate(refine_configs, 1):
            if cfg.config_id in done_ref:
                session.log(f"[refine {i}/{len(refine_configs)}] {cfg.config_id} — já concluído, pulando")
                continue
            refine_timer.mark_step_start()
            session.log(f"[refine {i}/{len(refine_configs)}] {cfg.config_id}")
            result = evaluate_search_config(
                cfg,
                base_config,
                base_train,
                dev_dataset,
                verbose=verbose,
                skip_existing=skip_existing or resume,
                enable_pruning=enable_pruning,
                pruning_warmup_epochs=pruning_warmup_epochs,
                pruning_margin=pruning_margin,
                pruning_min_folds=pruning_min_folds,
                global_best_mae_cm=global_best_mae,
                device=device,
                num_workers=num_workers,
            )
            refine_results.append(result)
            all_results.append(result)
            session.append_results_csv("hyperparams_refined_results.csv", result.to_row())
            if result.status == "completed":
                global_best_mae = min(global_best_mae or result.mae_mean_cm, result.mae_mean_cm)
            refine_timer.mark_step_end()
            refine_timer.step_done()
            session.log(
                f"  → MAE={result.mae_mean_cm:.4f} cm | score={result.composite:.4f} | "
                f"{refine_timer.status()} | {search_timer.status()}"
            )

        refine_results = _merge_phase_results(
            refine_results,
            load_phase_results_from_csv(session.run_dir, "hyperparams_refined_results.csv", "refine"),
        )

    # --- Fase C: estabilidade por sementes ---
    stability_results: list[ConfigResult] = []
    if stability_seeds:
        pool = sorted(refine_results or fast_results, key=lambda r: r.composite)
        top_cfgs = [r.config for r in pool[: max(1, refine_top_k or 1)]]
        if top_cfgs:
            stab_configs = generate_stability_configs(top_cfgs)
            session.log(f"Fase stability: {len(stab_configs)} sementes")
            done_stab = (
                session.load_completed_config_ids("hyperparams_stability_results.csv")
                if resume
                else set()
            )
            if resume:
                stability_results = load_phase_results_from_csv(
                    session.run_dir, "hyperparams_stability_results.csv", "stability"
                )
                seed_maes = [r.mae_mean_cm for r in stability_results]
                seeds_used = [r.config.seed for r in stability_results]
            stab_timer = RunTimer("stability")
            stab_timer.set_total(sum(1 for c in stab_configs if c.config_id not in done_stab))
            if not resume:
                seed_maes = []
            for j, cfg in enumerate(stab_configs, 1):
                if cfg.config_id in done_stab:
                    session.log(
                        f"[stability {j}/{len(stab_configs)}] seed={cfg.seed} "
                        f"{cfg.config_id} — já concluído, pulando"
                    )
                    continue
                seeds_used.append(cfg.seed)
                stab_timer.mark_step_start()
                session.log(f"[stability {j}/{len(stab_configs)}] seed={cfg.seed} {cfg.config_id}")
                result = evaluate_search_config(
                    cfg,
                    base_config,
                    base_train,
                    dev_dataset,
                    verbose=verbose,
                    skip_existing=skip_existing or resume,
                    device=device,
                    num_workers=num_workers,
                )
                seed_maes.append(result.mae_mean_cm)
                stability_results.append(result)
                row = result.to_row()
                row["seed_mae_mean"] = float(np.mean(seed_maes))
                row["seed_mae_std"] = float(np.std(seed_maes, ddof=1)) if len(seed_maes) > 1 else 0.0
                session.append_results_csv("hyperparams_stability_results.csv", row)
                stab_timer.mark_step_end()
                stab_timer.step_done()
                session.log(
                    f"  → MAE={result.mae_mean_cm:.4f} cm | "
                    f"{stab_timer.status()} | {search_timer.status()}"
                )

            if seed_maes:
                penalty = 0.05 * float(np.std(seed_maes, ddof=1)) if len(seed_maes) > 1 else 0.0
                for r in stability_results:
                    r.seed_mae_mean = float(np.mean(seed_maes))
                    r.seed_mae_std = float(np.std(seed_maes, ddof=1)) if len(seed_maes) > 1 else 0.0
                    r.composite = composite_score(
                        r.mae_mean_cm,
                        r.mae_std_cm,
                        r.rmse_mean_cm,
                        r.bias_cm,
                        r.proportional_bias_slope,
                        seed_stability_penalty=penalty,
                    )

    # --- Ranking final ---
    ranking_pool = stability_results or refine_results or fast_results
    ranking_sorted = sorted(ranking_pool, key=lambda r: r.composite)
    best = ranking_sorted[0]

    ranking_rows = [r.to_row() for r in ranking_sorted]
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_path = session.run_dir / "ranking_final.csv"
    ranking_df.to_csv(ranking_path, index=False)
    metrics_ranking = paths["metrics"] / "ranking_final.csv"
    ranking_df.to_csv(metrics_ranking, index=False)

    best_payload = build_best_config_payload(
        best,
        backend=backend,
        stability_seeds=stability_seeds,
        seeds_used=seeds_used if stability_seeds else None,
        base_train=base_train,
    )
    best_path = session.run_dir / "best_config.json"
    best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    session.copy_best_to_configs(best_path, paths["configs"])

    # Compatibilidade com etapa15 (best_hyperparameters.json)
    legacy = {
        "selected_at": best_payload["selected_at"],
        "selection_criterion": best_payload["selection_criterion"],
        "best_trial_id": best.config.config_id,
        "best_description": best.config.description,
        "mean_mae": best.mae_mean_cm,
        "std_mae": best.mae_std_cm,
        "experiment": best.experiment,
        "training": best.training,
    }
    (paths["configs"] / "best_hyperparameters.json").write_text(
        json.dumps(legacy, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    session.log(f"Melhor config: {best.config.config_id} | MAE={best.mae_mean_cm:.4f} cm | score={best.composite:.4f}")
    session.log(f"⏱ Busca concluída em {format_duration(search_timer.elapsed())}")
    (session.run_dir / "search_timing.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": round(search_timer.elapsed(), 2),
                "elapsed_human": format_duration(search_timer.elapsed()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    session.close()

    return all_results, best, session
