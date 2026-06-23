#!/usr/bin/env python3
"""
Etapa 14 — Escolha de hiperparâmetros (via LOSO nos 70%)
========================================================
Compara combinações de hiperparâmetros usando LOSO no grupo de
desenvolvimento. Escolhe a melhor configuração pela MAE média entre folds.

NÃO usa teste final (30%). NÃO roda automaticamente — execute quando quiser.

Como rodar no futuro
--------------------
    cd /Users/Rodacki/Desktop/Hoffmann/UTT

    # listar trials do grid
    .venv/bin/python pipeline/etapa14_hyperparameters.py --list

    # busca completa (lenta: trials × 19 folds)
    .venv/bin/python pipeline/etapa14_hyperparameters.py --run

    # triagem rápida (3 sujeitos dev por trial)
    .venv/bin/python pipeline/etapa14_hyperparameters.py --run --quick-folds 3

    # retomar trials já concluídos
    .venv/bin/python pipeline/etapa14_hyperparameters.py --run --skip-existing

    # rodar só alguns trials
    .venv/bin/python pipeline/etapa14_hyperparameters.py --run --trials tcn_baseline,cnn1d

Tempo estimado (--run completo, 8 trials × 19 folds): várias horas em CPU.
Use --quick-folds 3 para triagem inicial (~30–60 min).
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from etapa01_setup import ExperimentConfig, build_default_config, create_output_dirs
from etapa11_loss_metrics import LossName, TrainingConfig
from etapa13_loso import load_dev_dataset, run_loso_on_dev

ModelType = Literal["tcn", "cnn1d"]


# ---------------------------------------------------------------------------
# 1. Definição de um trial de hiperparâmetros
# ---------------------------------------------------------------------------
@dataclass
class HyperparamTrial:
    """
    Combinação de hiperparâmetros a avaliar via LOSO.

    Campos omitidos mantêm o padrão de ExperimentConfig / TrainingConfig.
    """

    trial_id: str
    description: str
    model_type: ModelType | None = None
    window_seconds: float | None = None
    stride_seconds: float | None = None
    loss: LossName | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    weight_decay: float | None = None
    dropout: float | None = None
    max_epochs: int | None = None
    patience: int | None = None

    def artifact_prefix(self) -> str:
        return f"etapa14_{self.trial_id}"


@dataclass
class TrialResult:
    trial_id: str
    description: str
    n_folds: int
    mean_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    experiment: dict[str, Any]
    training: dict[str, Any]
    summary_path: Path


# ---------------------------------------------------------------------------
# 2. Grid de busca (curado — expanda conforme necessário)
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_GRID: list[HyperparamTrial] = [
    HyperparamTrial(
        trial_id="tcn_baseline",
        description="TCN | janela 2s | stride 1s | Huber | lr 1e-3",
        model_type="tcn",
        window_seconds=2.0,
        stride_seconds=1.0,
        loss="huber",
        learning_rate=1e-3,
        batch_size=32,
    ),
    HyperparamTrial(
        trial_id="cnn1d",
        description="CNN 1D | janela 2s | stride 1s | Huber | lr 1e-3",
        model_type="cnn1d",
        window_seconds=2.0,
        stride_seconds=1.0,
        loss="huber",
        learning_rate=1e-3,
        batch_size=32,
    ),
    HyperparamTrial(
        trial_id="win_1s",
        description="TCN | janela 1s | stride 0.5s",
        model_type="tcn",
        window_seconds=1.0,
        stride_seconds=0.5,
    ),
    HyperparamTrial(
        trial_id="win_3s",
        description="TCN | janela 3s | stride 1.5s",
        model_type="tcn",
        window_seconds=3.0,
        stride_seconds=1.5,
    ),
    HyperparamTrial(
        trial_id="loss_mse",
        description="TCN | loss MSE",
        model_type="tcn",
        loss="mse",
    ),
    HyperparamTrial(
        trial_id="lr_5e4",
        description="TCN | learning rate 5e-4",
        model_type="tcn",
        learning_rate=5e-4,
    ),
    HyperparamTrial(
        trial_id="batch_16",
        description="TCN | batch size 16",
        model_type="tcn",
        batch_size=16,
    ),
    HyperparamTrial(
        trial_id="stride_05",
        description="TCN | stride 0.5s (75% sobreposição)",
        model_type="tcn",
        stride_seconds=0.5,
    ),
]


HYPERPARAM_GUIDE: dict[str, str] = {
    "window_seconds": "Contexto temporal. Curta → padrões locais; longa → ciclo completo do movimento.",
    "stride_seconds": "Passo entre janelas. Menor → mais janelas, mais correlação entre amostras.",
    "model_type": "TCN enxerga contexto longo; CNN 1D é baseline mais simples.",
    "loss": "Huber robusta a outliers; MSE penaliza erros grandes; MAE mais robusta porém menos suave.",
    "learning_rate": "Passo do Adam. Alto demais → instável; baixo demais → converge devagar.",
    "batch_size": "Maior → gradiente mais estável; menor → mais ruído, às vezes generaliza melhor.",
    "weight_decay": "Regularização L2 — reduz overfitting em modelos maiores (TCN).",
}


# ---------------------------------------------------------------------------
# 3. Aplicar trial às configs
# ---------------------------------------------------------------------------
def apply_trial(
    base_config: ExperimentConfig,
    base_train: TrainingConfig,
    trial: HyperparamTrial,
) -> tuple[ExperimentConfig, TrainingConfig]:
    """Cria cópias de config com overrides do trial."""
    config = deepcopy(base_config)
    train = deepcopy(base_train)

    if trial.model_type is not None:
        config.model_type = trial.model_type
    if trial.window_seconds is not None:
        config.window_seconds = trial.window_seconds
    if trial.stride_seconds is not None:
        config.stride_seconds = trial.stride_seconds
    if trial.loss is not None:
        train.loss = trial.loss
    if trial.learning_rate is not None:
        train.learning_rate = trial.learning_rate
    if trial.batch_size is not None:
        train.batch_size = trial.batch_size
    if trial.weight_decay is not None:
        train.weight_decay = trial.weight_decay
    if trial.dropout is not None:
        train.dropout = trial.dropout
    if trial.max_epochs is not None:
        train.max_epochs = trial.max_epochs
    if trial.patience is not None:
        train.patience = trial.patience

    return config, train


def configs_to_dict(config: ExperimentConfig, train: TrainingConfig) -> tuple[dict, dict]:
    exp = {
        "model_type": config.model_type,
        "window_seconds": config.window_seconds,
        "stride_seconds": config.stride_seconds,
        "window_samples": config.window_samples,
        "stride_samples": config.stride_samples,
        "target_mode": config.target_mode,
        "sampling_hz": config.sampling_hz,
    }
    tr = asdict(train)
    return exp, tr


# ---------------------------------------------------------------------------
# 4. Avaliar um trial via LOSO
# ---------------------------------------------------------------------------
def trial_summary_path(metrics_dir: Path, trial_id: str) -> Path:
    return metrics_dir / f"etapa14_{trial_id}_loso_summary.json"


def is_trial_completed(metrics_dir: Path, trial_id: str) -> bool:
    return trial_summary_path(metrics_dir, trial_id).is_file()


def evaluate_trial(
    trial: HyperparamTrial,
    base_config: ExperimentConfig,
    base_train: TrainingConfig,
    dev_dataset,
    subject_ids: list[str] | None = None,
    verbose: bool = False,
    skip_existing: bool = False,
) -> TrialResult:
    """Executa LOSO para um trial e retorna métricas agregadas."""
    config, train_cfg = apply_trial(base_config, base_train, trial)
    paths = create_output_dirs(config.output_dir)
    prefix = trial.artifact_prefix()

    if skip_existing and is_trial_completed(paths["metrics"], trial.trial_id):
        payload = json.loads(trial_summary_path(paths["metrics"], trial.trial_id).read_text(encoding="utf-8"))
        agg = payload["aggregate"]
        exp, tr = configs_to_dict(config, train_cfg)
        print(f"  → trial '{trial.trial_id}' já concluído — MAE={agg['mae_mean']:.4f}\n")
        return TrialResult(
            trial_id=trial.trial_id,
            description=trial.description,
            n_folds=int(payload["n_folds"]),
            mean_mae=float(agg["mae_mean"]),
            std_mae=float(agg["mae_std"]),
            mean_rmse=float(agg["rmse_mean"]),
            std_rmse=float(agg["rmse_std"]),
            mean_r2=float(agg["r2_mean"]),
            experiment=exp,
            training=tr,
            summary_path=trial_summary_path(paths["metrics"], trial.trial_id),
        )

    print(f"Trial: {trial.trial_id} — {trial.description}")
    exp, tr = configs_to_dict(config, train_cfg)
    print(f"  model={exp['model_type']} | janela={exp['window_seconds']}s | stride={exp['stride_seconds']}s | loss={tr['loss']}")

    _, summary, pruned = run_loso_on_dev(
        config=config,
        train_cfg=train_cfg,
        dev_dataset=dev_dataset,
        subject_ids=subject_ids,
        verbose=verbose,
        save_individual_plots=False,
        skip_existing=skip_existing,
        artifact_prefix=prefix,
    )

    return TrialResult(
        trial_id=trial.trial_id,
        description=trial.description,
        n_folds=summary.n_folds,
        mean_mae=summary.mean_mae,
        std_mae=summary.std_mae,
        mean_rmse=summary.mean_rmse,
        std_rmse=summary.std_rmse,
        mean_r2=summary.mean_r2,
        experiment=exp,
        training=tr,
        summary_path=trial_summary_path(paths["metrics"], trial.trial_id),
    )


def select_best_trial(results: list[TrialResult]) -> TrialResult:
    """Escolhe trial com menor MAE média (critério principal)."""
    return min(results, key=lambda r: r.mean_mae)


# ---------------------------------------------------------------------------
# 5. Busca e persistência
# ---------------------------------------------------------------------------
def run_hyperparameter_search(
    trials: list[HyperparamTrial],
    base_config: ExperimentConfig | None = None,
    base_train: TrainingConfig | None = None,
    quick_folds: int | None = None,
    verbose: bool = False,
    skip_existing: bool = False,
) -> tuple[list[TrialResult], TrialResult]:
    """
    Avalia cada trial via LOSO e seleciona o melhor.

    quick_folds : se definido, usa apenas os N primeiros sujeitos dev (triagem).
    """
    if base_config is None:
        base_config = build_default_config()
    if base_train is None:
        base_train = TrainingConfig()

    dev_dataset = load_dev_dataset(base_config)
    all_ids = dev_dataset.get_subject_ids()
    subject_ids = all_ids[:quick_folds] if quick_folds else all_ids

    paths = create_output_dirs(base_config.output_dir)

    print("=" * 60)
    print("ETAPA 14 — Busca de hiperparâmetros (LOSO nos 70%)")
    print("=" * 60)
    print(f"Trials          : {len(trials)}")
    print(f"Folds por trial : {len(subject_ids)} sujeito(s) dev")
    if quick_folds:
        print(f"Modo triagem    : --quick-folds {quick_folds} (NÃO é avaliação final)")
    print("Teste final 30% : NÃO utilizado\n")

    results: list[TrialResult] = []
    for i, trial in enumerate(trials, start=1):
        print(f"[{i}/{len(trials)}] {trial.trial_id}")
        result = evaluate_trial(
            trial,
            base_config,
            base_train,
            dev_dataset=dev_dataset,
            subject_ids=subject_ids,
            verbose=verbose,
            skip_existing=skip_existing,
        )
        results.append(result)
        print(
            f"  → MAE={result.mean_mae:.4f} ± {result.std_mae:.4f} | "
            f"RMSE={result.mean_rmse:.4f}\n"
        )

    best = select_best_trial(results)
    _save_search_results(results, best, paths, quick_folds=quick_folds)
    print_best_trial_report(results, best, paths, quick_folds=quick_folds)
    return results, best


def _save_search_results(
    results: list[TrialResult],
    best: TrialResult,
    paths: dict[str, Path],
    quick_folds: int | None,
) -> None:
    rows = [
        {
            "trial_id": r.trial_id,
            "description": r.description,
            "n_folds": r.n_folds,
            "mean_mae": r.mean_mae,
            "std_mae": r.std_mae,
            "mean_rmse": r.mean_rmse,
            "std_rmse": r.std_rmse,
            "mean_r2": r.mean_r2,
            "is_best": r.trial_id == best.trial_id,
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("mean_mae")
    csv_path = paths["metrics"] / "etapa14_search_results.csv"
    df.to_csv(csv_path, index=False)

    best_payload = {
        "selected_at": datetime.now().isoformat(timespec="seconds"),
        "selection_criterion": "menor mean_mae no LOSO dev",
        "quick_folds_mode": quick_folds,
        "note": (
            "Se quick_folds foi usado, revalide o trial vencedor com LOSO completo "
            "antes da Etapa 15."
        ),
        "best_trial_id": best.trial_id,
        "best_description": best.description,
        "mean_mae": best.mean_mae,
        "std_mae": best.std_mae,
        "experiment": best.experiment,
        "training": best.training,
    }
    best_path = paths["configs"] / "best_hyperparameters.json"
    best_path.write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ranking_path = paths["metrics"] / "etapa14_search_ranking.json"
    ranking_path.write_text(
        json.dumps({"trials": rows, "best_trial_id": best.trial_id}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_best_trial_report(
    results: list[TrialResult],
    best: TrialResult,
    paths: dict[str, Path],
    quick_folds: int | None,
) -> None:
    print("\nRanking (menor MAE primeiro):")
    print(f"  {'trial_id':<16} {'MAE':>10} {'±std':>8} {'RMSE':>10}")
    print("  " + "-" * 48)
    for r in sorted(results, key=lambda x: x.mean_mae):
        mark = " ← melhor" if r.trial_id == best.trial_id else ""
        print(f"  {r.trial_id:<16} {r.mean_mae:10.4f} {r.std_mae:8.4f} {r.mean_rmse:10.4f}{mark}")

    print(f"\nMelhor trial: {best.trial_id}")
    print(f"  {best.description}")
    print(f"  MAE  = {best.mean_mae:.4f} ± {best.std_mae:.4f}")
    print(f"  RMSE = {best.mean_rmse:.4f} ± {best.std_rmse:.4f}")

    if quick_folds:
        print("\n⚠ Modo triagem (--quick-folds): revalide o vencedor com LOSO completo:")
        print(f"  .venv/bin/python pipeline/etapa14_hyperparameters.py --run --trials {best.trial_id}")

    print("\nArquivos salvos:")
    print(f"  {paths['metrics'] / 'etapa14_search_results.csv'}")
    print(f"  {paths['configs'] / 'best_hyperparameters.json'}")
    print("\nPróxima etapa: treino final nos 70% + teste nos 30% (Etapa 15).")
    print("=" * 60)


def print_trial_list(trials: list[HyperparamTrial]) -> None:
    print("=" * 60)
    print("ETAPA 14 — Grid de hiperparâmetros (LOSO nos 70%)")
    print("=" * 60)
    print(f"Trials disponíveis: {len(trials)}\n")
    for t in trials:
        print(f"  {t.trial_id:<16} {t.description}")

    print("\nO que cada eixo controla:")
    for key, desc in HYPERPARAM_GUIDE.items():
        print(f"  {key:<16} {desc}")

    print("\nCritério de seleção: menor MAE média no LOSO dev.")
    print("Teste final (30%): intocado até a Etapa 15.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 14 — escolha de hiperparâmetros via LOSO (70% dev, sem teste final)."
    )
    parser.add_argument("--list", action="store_true", help="listar trials do grid e sair")
    parser.add_argument("--run", action="store_true", help="executar busca de hiperparâmetros")
    parser.add_argument(
        "--trials",
        type=str,
        default="",
        help="IDs separados por vírgula (ex.: tcn_baseline,cnn1d). Vazio = grid completo.",
    )
    parser.add_argument(
        "--quick-folds",
        type=int,
        default=None,
        metavar="N",
        help="triagem rápida com N sujeitos dev por trial",
    )
    parser.add_argument("--verbose", action="store_true", help="log de épocas por fold")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="pular trials/folds cujos resumos LOSO já existem",
    )
    return parser.parse_args()


def _filter_trials(all_trials: list[HyperparamTrial], ids_csv: str) -> list[HyperparamTrial]:
    if not ids_csv.strip():
        return all_trials
    wanted = {x.strip() for x in ids_csv.split(",") if x.strip()}
    selected = [t for t in all_trials if t.trial_id in wanted]
    missing = wanted - {t.trial_id for t in selected}
    if missing:
        raise ValueError(f"Trial(s) desconhecido(s): {sorted(missing)}")
    return selected


if __name__ == "__main__":
    args = parse_args()
    grid = DEFAULT_SEARCH_GRID

    if args.list or not args.run:
        print_trial_list(grid)
        if not args.run:
            print("\nPara executar a busca: adicione --run")
    else:
        trials = _filter_trials(grid, args.trials)
        run_hyperparameter_search(
            trials=trials,
            quick_folds=args.quick_folds,
            verbose=args.verbose,
            skip_existing=args.skip_existing,
        )
