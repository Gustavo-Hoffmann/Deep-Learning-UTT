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
    else:
        m = metrics_from_cm_arrays(y_true_cm, y_pred_cm)
        mae_mean = m.mae
        rmse_mean = m.rmse
        bias = m.bias
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
        bias_cm=bias,
        proportional_bias_slope=slope,
        composite=comp,
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
    stability_seeds: bool = False,
    resume: bool = False,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    verbose: bool = False,
    skip_existing: bool = False,
    device=None,
    num_workers: int = 0,
) -> tuple[list[ConfigResult], ConfigResult, RunSession]:
    """Executa busca progressiva: fast → refine → stability."""
    if base_config is None:
        base_config = build_default_config()
    if base_train is None:
        base_train = TrainingConfig()

    session = RunSession(base_config.output_dir, resume=resume)
    session.open_log()
    session.save_config(
        {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "refine_top_k": refine_top_k,
            "stability_seeds": stability_seeds,
            "pruning": enable_pruning,
            "training": asdict(base_train),
        }
    )

    dev_dataset = load_dev_dataset(base_config)
    all_results: list[ConfigResult] = []
    tested_fps: set[str] = set()
    global_best_mae: float | None = None

    # --- Fase A: busca rápida ---
    fast_configs = generate_fast_grid()
    session.log(f"Fase fast: {len(fast_configs)} configurações")
    done_fast = session.load_completed_config_ids("hyperparams_fast_results.csv") if resume else set()

    for i, cfg in enumerate(fast_configs, 1):
        if cfg.config_id in done_fast:
            session.log(f"[fast {i}/{len(fast_configs)}] {cfg.config_id} — já concluído, pulando")
            continue
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
            global_best_mae_cm=global_best_mae,
            device=device,
            num_workers=num_workers,
        )
        all_results.append(result)
        session.append_results_csv("hyperparams_fast_results.csv", result.to_row())
        if result.status == "completed":
            global_best_mae = result.mae_mean_cm if global_best_mae is None else min(global_best_mae, result.mae_mean_cm)
        session.log(f"  → MAE={result.mae_mean_cm:.4f} cm | score={result.composite:.4f}")

    fast_results = [r for r in all_results if r.config.phase == "fast"]
    fast_sorted = sorted(fast_results, key=lambda r: r.composite)

    # --- Fase B: refinamento ---
    refine_results: list[ConfigResult] = []
    if refine_top_k > 0 and fast_sorted:
        top = [r.config for r in fast_sorted[:refine_top_k]]
        for r in fast_results:
            tested_fps.add(r.config.fingerprint())
        refine_configs = generate_refine_grid(top, tested_fps)
        session.log(f"Fase refine: {len(refine_configs)} configurações (top-{refine_top_k})")
        done_ref = session.load_completed_config_ids("hyperparams_refined_results.csv") if resume else set()

        for i, cfg in enumerate(refine_configs, 1):
            if cfg.config_id in done_ref:
                continue
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
                global_best_mae_cm=global_best_mae,
                device=device,
                num_workers=num_workers,
            )
            refine_results.append(result)
            all_results.append(result)
            session.append_results_csv("hyperparams_refined_results.csv", result.to_row())
            if result.status == "completed":
                global_best_mae = min(global_best_mae or result.mae_mean_cm, result.mae_mean_cm)

    # --- Fase C: estabilidade por sementes ---
    stability_results: list[ConfigResult] = []
    if stability_seeds:
        pool = sorted(refine_results or fast_results, key=lambda r: r.composite)
        top_cfgs = [r.config for r in pool[: max(1, refine_top_k or 1)]]
        if top_cfgs:
            stab_configs = generate_stability_configs(top_cfgs)
            session.log(f"Fase stability: {len(stab_configs)} sementes")
            seed_maes: list[float] = []
            for cfg in stab_configs:
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

    best_payload = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "selection_criterion": "composite_score (MAE + std + bias + slope + RMSE)",
        "config_id": best.config.config_id,
        "composite_score": best.composite,
        "mae_mean_cm": best.mae_mean_cm,
        "rmse_mean_cm": best.rmse_mean_cm,
        "bias_cm": best.bias_cm,
        "proportional_bias_slope": best.proportional_bias_slope,
        "experiment": best.experiment,
        "training": best.training,
        "per_fold_best_epochs": best.per_fold_best_epochs,
        "final_epochs_p75": int(np.percentile(best.per_fold_best_epochs, 75)) if best.per_fold_best_epochs else None,
    }
    best_path = session.run_dir / "best_config.json"
    best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    paths = create_output_dirs(base_config.output_dir)
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
    session.close()
    return all_results, best, session
