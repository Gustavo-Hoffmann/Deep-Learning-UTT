#!/usr/bin/env python3
"""
Pipeline completo — IMU smartphone → amplitude Vicon (Etapas 1–16)
==================================================================
Um único ponto de entrada: carrega os dados, valida, divide sujeitos,
treina (LOSO + teste final) e gera gráficos.

Uso recomendado (CPU robusto):
    python run_pipeline.py --device cpu --cpu-full-throttle --diagnostics \\
        --baselines --with-hyperparams-fast --refine-top-k 5 \\
        --stability-seeds --pruning --calibrate-from-loso --max-epochs 300 --patience 30

Retomar busca interrompida:
    python run_pipeline.py --device cpu --cpu-full-throttle --resume \\
        --with-hyperparams-fast --refine-top-k 5 --stability-seeds --pruning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from etapa01_setup import (  # noqa: E402
    ExperimentConfig,
    RuntimeConfig,
    build_default_config,
    create_output_dirs,
    get_device,
    run_stage01_setup,
    setup_runtime,
)
from etapa02_schema import run_stage02_schema  # noqa: E402
from etapa03_load import LoadedDataset, run_stage03_load  # noqa: E402
from etapa04_quality import run_stage04_quality  # noqa: E402
from etapa05_split import run_stage05_split  # noqa: E402
from etapa11_loss_metrics import TrainingConfig  # noqa: E402
from etapa13_loso import run_stage13_loso  # noqa: E402
from etapa14_hyperparameters import DEFAULT_SEARCH_GRID, run_hyperparameter_search  # noqa: E402
from etapa14_progressive_search import run_progressive_search  # noqa: E402
from etapa15_final_test import infer_epochs_from_loso, run_stage15_final_test  # noqa: E402
from etapa16_visualize import run_stage16_visualize  # noqa: E402
from etapa_baselines import run_baselines  # noqa: E402
from etapa_diagnostics import run_diagnostics  # noqa: E402
from etapa_evaluation import plot_bland_altman  # noqa: E402

HyperparamMode = Literal["skip", "quick", "fast", "full"]


@dataclass
class PipelineResult:
    config: ExperimentConfig
    runtime: RuntimeConfig | None
    dataset: LoadedDataset | None
    loso_mae: float | None
    test_mae_cm: float | None
    test_mae_cm_calibrated: float | None
    plot_paths: list[Path]
    elapsed_seconds: float
    run_session_dir: Path | None = None
    flags: dict = field(default_factory=dict)


def _banner(title: str, step: int, total: int = 16) -> None:
    print("\n" + "=" * 70)
    print(f"[{step}/{total}] {title}")
    print("=" * 70)


def _format_duration(seconds: float) -> str:
    """Formata segundos como '2h 15m 30s', '45m 12s' ou '38s'."""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _save_run_timing(
    paths: dict[str, Path],
    *,
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """Persiste duração do run para consulta posterior."""
    finished_at = datetime.now()
    payload = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_human": _format_duration(elapsed_seconds),
    }
    timing_path = paths["configs"] / "run_timing.json"
    timing_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    config_path = paths["configs"] / "config.json"
    if config_path.is_file():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["finished_at"] = payload["finished_at"]
        config_payload["elapsed_seconds"] = payload["elapsed_seconds"]
        config_payload["elapsed_human"] = payload["elapsed_human"]
        config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return timing_path


def _parse_hyperparams_mode(args: argparse.Namespace) -> HyperparamMode:
    if args.all:
        return "full"
    if args.with_hyperparams_fast or args.with_hyperparams:
        return "fast"
    return "skip"


def _parse_final_epochs(args: argparse.Namespace, metrics_dir: Path) -> int | None:
    if args.final_epochs is not None:
        return int(args.final_epochs)
    if args.epochs.lower() != "auto":
        return int(args.epochs)
    return None


def _clear_loso_artifacts(output_dir: Path) -> int:
    subdirs = ("metrics", "checkpoints", "plots", "scalers")
    patterns = ("etapa13_*", "etapa16_loso_*")
    removed = 0
    for sub in subdirs:
        folder = output_dir / sub
        if not folder.is_dir():
            continue
        for pattern in patterns:
            for path in folder.glob(pattern):
                if path.is_file():
                    path.unlink()
                    removed += 1
    return removed


def run_full_pipeline(
    *,
    hyperparams: HyperparamMode = "skip",
    skip_existing: bool = False,
    resume: bool = False,
    verbose: bool = False,
    n_epochs: int | None = None,
    quality_plots: bool = True,
    loso_plots: bool = False,
    from_stage: int = 1,
    runtime: RuntimeConfig | None = None,
    device_obj=None,
    train_cfg: TrainingConfig | None = None,
    diagnostics: bool = False,
    baselines: bool = False,
    refine_top_k: int = 0,
    stability_seeds: bool = False,
    enable_pruning: bool = False,
    pruning_warmup_epochs: int = 30,
    pruning_margin: float = 1.50,
    calibrate_from_loso: bool = False,
) -> PipelineResult:
    t0 = time.perf_counter()
    started_at = datetime.now()
    print("\n" + "⏱" * 3 + " CRONÔMETRO DO RUN " + "⏱" * 3)
    print(f"Início  : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Duração : será exibida ao final do pipeline\n")

    config: ExperimentConfig | None = None
    dataset: LoadedDataset | None = None
    loso_mae: float | None = None
    test_mae_cm: float | None = None
    test_mae_cm_cal: float | None = None
    plot_paths: list[Path] = []
    run_session_dir: Path | None = None
    flags = {
        "fast_search": hyperparams == "fast",
        "refine": refine_top_k > 0,
        "stability": stability_seeds,
        "pruning": enable_pruning,
        "baselines": baselines,
        "diagnostics": diagnostics,
        "calibration": calibrate_from_loso,
    }

    if train_cfg is None:
        train_cfg = TrainingConfig()

    use_best_hparams = hyperparams != "skip"

    # ------------------------------------------------------------------ 1
    if from_stage <= 1:
        _banner("Preparação do ambiente", 1)
        if runtime is not None and device_obj is not None:
            config = build_default_config(PROJECT_ROOT)
            paths = create_output_dirs(config.output_dir)
            from etapa01_setup import save_config_snapshot, set_seed, SEED

            set_seed(SEED)
            save_config_snapshot(config, paths, runtime=runtime)
            print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if runtime.device_request == "cpu":
                print("Dispositivo selecionado: CPU forçado pelo usuário")
            print(f"Dispositivo : {device_obj} ({runtime.device_description})")
            if runtime.num_threads:
                print(f"Threads CPU : {runtime.num_threads}")
        else:
            config, paths, device_obj, runtime = run_stage01_setup(PROJECT_ROOT)
            print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        config = build_default_config(PROJECT_ROOT)
        paths = create_output_dirs(config.output_dir)
        if device_obj is None:
            device_obj = get_device()
        if runtime is None:
            _, runtime = setup_runtime()

    assert config is not None
    train_cfg.num_workers = runtime.num_workers if runtime else 0

    # ------------------------------------------------------------------ 2
    if from_stage <= 2:
        _banner("Contrato dos dados (schema)", 2)
        run_stage02_schema(config)

    # ------------------------------------------------------------------ 3
    if from_stage <= 3:
        _banner("Leitura dos arquivos por sujeito", 3)
        dataset = run_stage03_load(config)
    elif from_stage <= 5:
        dataset = run_stage03_load(config)

    # ------------------------------------------------------------------ 4
    if from_stage <= 4:
        _banner("Qualidade e alinhamento dos sinais", 4)
        if dataset is None:
            dataset = run_stage03_load(config)
        run_stage04_quality(
            config,
            dataset=dataset,
            max_example_plots=3 if quality_plots else 0,
        )

    # ------------------------------------------------------------------ 5
    if from_stage <= 5:
        _banner("Split externo 70/30 por sujeito", 5)
        if dataset is None:
            dataset = run_stage03_load(config)
        run_stage05_split(config, dataset=dataset)

    # ------------------------------------------------------------------ Diagnósticos
    if diagnostics and from_stage <= 6:
        _banner("Diagnósticos de dados", 6)
        run_diagnostics(config)

    # ------------------------------------------------------------------ Baselines
    if baselines and from_stage <= 12:
        _banner("Baselines (LOSO 70%)", 12)
        run_baselines(config)

    # ------------------------------------------------------------------ 6–12
    if from_stage <= 13:
        print("\n" + "-" * 70)
        print("Etapas 6–12: executadas dentro do LOSO (normalização, janelas,")
        print("modelo, loss e treino por fold — sem passo separado aqui).")
        print("-" * 70)

    # ------------------------------------------------------------------ 13
    if from_stage <= 13:
        _banner("LOSO completo nos 70% de desenvolvimento", 13)
        if not skip_existing and not resume:
            n_cleared = _clear_loso_artifacts(config.output_dir)
            if n_cleared:
                print(f"Artefatos anteriores da Etapa 13 removidos ({n_cleared} arquivo(s)).")
            print("LOSO: treino do zero.\n")
        else:
            print("LOSO: retomando folds já concluídos.\n")

        _, loso_summary = run_stage13_loso(
            config=config,
            train_cfg=train_cfg,
            verbose=verbose,
            save_individual_plots=loso_plots,
            skip_existing=skip_existing or resume,
            device=device_obj,
        )
        loso_mae = loso_summary.mean_mae

        # Bland-Altman LOSO OOF
        oof_path = paths["predictions"] / "etapa13_oof_predictions.csv"
        if oof_path.is_file():
            import pandas as pd

            oof = pd.read_csv(oof_path)
            plot_bland_altman(
                oof["y_true_cm"].to_numpy(),
                oof["y_pred_cm"].to_numpy(),
                paths["plots"] / "etapa13_bland_altman_loso.png",
                "Bland-Altman LOSO OOF (70% dev)",
                subject_ids=oof["subject_id"].to_numpy(),
            )
    elif from_stage > 13:
        summary_path = paths["metrics"] / "etapa13_loso_summary.json"
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            loso_mae = payload.get("aggregate", {}).get("mae_mean")
            print(f"LOSO existente carregado: MAE média = {loso_mae:.4f}\n")

    # ------------------------------------------------------------------ 14
    if from_stage <= 14 and hyperparams == "fast":
        _banner("Busca progressiva de hiperparâmetros (LOSO dev)", 14)
        print("Modo: busca rápida → refinamento → estabilidade (se solicitado)\n")
        _, best, session = run_progressive_search(
            base_config=config,
            base_train=train_cfg,
            refine_top_k=refine_top_k,
            stability_seeds=stability_seeds,
            resume=resume,
            enable_pruning=enable_pruning,
            pruning_warmup_epochs=pruning_warmup_epochs,
            pruning_margin=pruning_margin,
            verbose=verbose,
            skip_existing=skip_existing or resume,
            device=device_obj,
            num_workers=runtime.num_workers if runtime else 0,
        )
        run_session_dir = session.run_dir
        print(f"Melhor config: {best.config.config_id} (MAE={best.mae_mean_cm:.4f} cm, score={best.composite:.4f})")
        use_best_hparams = True
    elif from_stage <= 14 and hyperparams == "full":
        _banner("Busca de hiperparâmetros (grid legado)", 14)
        print("Aviso: --all bruto não é recomendado. Use busca progressiva (--with-hyperparams-fast).\n")
        trials = DEFAULT_SEARCH_GRID
        _, best = run_hyperparameter_search(
            trials=trials,
            base_config=config,
            base_train=train_cfg,
            quick_folds=None,
            verbose=verbose,
            skip_existing=skip_existing or resume,
        )
        print(f"Melhor trial: {best.trial_id} (MAE={best.mean_mae:.4f})")
        use_best_hparams = True
    elif from_stage <= 14:
        print("\n" + "-" * 70)
        print("Etapa 14: busca omitida (config padrão TCN).")
        print("Use --with-hyperparams-fast para busca progressiva.")
        print("-" * 70)
    elif hyperparams != "skip":
        use_best_hparams = True

    # ------------------------------------------------------------------ 15
    if from_stage <= 15:
        _banner("Treino final (70% dev) + teste intocado (30%)", 15)
        if n_epochs is None:
            n_epochs = infer_epochs_from_loso(paths["metrics"])
            print(f"Épocas finais (auto, P75 LOSO): {n_epochs}\n")

        metrics = run_stage15_final_test(
            config=config,
            train_cfg=train_cfg,
            n_epochs=n_epochs,
            use_best_hparams=use_best_hparams,
            verbose=verbose,
            calibrate_from_loso=calibrate_from_loso,
            device=device_obj,
            num_workers=runtime.num_workers if runtime else 0,
        )
        test_mae_cm = metrics["metrics_cm"]["mae"]
        if metrics.get("calibration"):
            test_mae_cm_cal = metrics["calibration"]["after_calibration"]["mae"]

        # Bland-Altman teste final
        import pandas as pd

        pred_path = paths["predictions"] / "etapa15_test_predictions.csv"
        if pred_path.is_file():
            pred_df = pd.read_csv(pred_path)
            plot_bland_altman(
                pred_df["y_true_cm"].to_numpy(),
                pred_df["y_pred_cm"].to_numpy(),
                paths["plots"] / "etapa16_test_bland_altman.png",
                "Bland-Altman — teste final (30%)",
                subject_ids=pred_df["subject_id"].to_numpy(),
            )

    # ------------------------------------------------------------------ 16
    if from_stage <= 16:
        _banner("Visualização dos resultados", 16)
        plot_paths = run_stage16_visualize(source="all")

    elapsed = time.perf_counter() - t0
    timing_path = _save_run_timing(paths, started_at=started_at, elapsed_seconds=elapsed)
    _print_final_summary(
        config,
        runtime,
        loso_mae,
        test_mae_cm,
        test_mae_cm_cal,
        plot_paths,
        elapsed,
        started_at,
        flags,
        run_session_dir,
        paths,
        timing_path,
    )
    return PipelineResult(
        config=config,
        runtime=runtime,
        dataset=dataset,
        loso_mae=loso_mae,
        test_mae_cm=test_mae_cm,
        test_mae_cm_calibrated=test_mae_cm_cal,
        plot_paths=plot_paths,
        elapsed_seconds=elapsed,
        run_session_dir=run_session_dir,
        flags=flags,
    )


def _print_final_summary(
    config: ExperimentConfig,
    runtime: RuntimeConfig | None,
    loso_mae: float | None,
    test_mae_cm: float | None,
    test_mae_cm_cal: float | None,
    plot_paths: list[Path],
    elapsed: float,
    started_at: datetime,
    flags: dict,
    run_session_dir: Path | None,
    paths: dict[str, Path],
    timing_path: Path,
) -> None:
    finished_at = datetime.now()
    print("\n" + "=" * 70)
    print("PIPELINE CONCLUÍDO — resumo final")
    print("=" * 70)
    print(f"Cronômetro             : {_format_duration(elapsed)} ({elapsed:.0f} s)")
    print(f"Início do run          : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Término do run         : {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if runtime:
        print(f"Dispositivo solicitado : {runtime.device_request}")
        print(f"Dispositivo efetivo    : {runtime.device_effective}")
        if runtime.device_request == "cpu":
            print("CPU forçada            : sim")
        print(f"Threads                : {runtime.num_threads}")
        print(f"Workers DataLoader     : {runtime.num_workers}")
    print(f"Modelo                 : {config.model_type.upper()}")
    print(f"Janela / stride        : {config.window_seconds}s / {config.stride_seconds}s")
    print(f"Saídas                 : {config.output_dir}")

    best_cfg = paths["configs"] / "best_config.json"
    if best_cfg.is_file():
        best = json.loads(best_cfg.read_text(encoding="utf-8"))
        print(f"Melhor configuração    : {best.get('config_id', 'N/A')}")
        print(f"  MAE LOSO (cm)        : {best.get('mae_mean_cm', 'N/A')}")
        print(f"  Score composto       : {best.get('composite_score', 'N/A')}")

    print(f"Busca rápida           : {'sim' if flags.get('fast_search') else 'não'}")
    print(f"Refinamento            : {'sim' if flags.get('refine') else 'não'}")
    print(f"Estabilidade (seeds)   : {'sim' if flags.get('stability') else 'não'}")
    print(f"Pruning                : {'sim' if flags.get('pruning') else 'não'}")
    print(f"Baselines              : {'sim' if flags.get('baselines') else 'não'}")
    print(f"Diagnósticos           : {'sim' if flags.get('diagnostics') else 'não'}")

    if loso_mae is not None:
        print(f"LOSO dev (MAE norm.)   : {loso_mae:.4f}")

    baseline_summary = paths["metrics"] / "baseline_summary.csv"
    if baseline_summary.is_file():
        import pandas as pd

        bs = pd.read_csv(baseline_summary)
        print("Baselines (MAE médio):")
        for _, row in bs.iterrows():
            print(f"  {row['baseline']:<16} {row['mae_mean']:.4f}")

    if test_mae_cm is not None:
        print(f"Teste final (MAE cm)   : {test_mae_cm:.4f} cm")
    if test_mae_cm_cal is not None:
        print(f"Teste pós-calibração   : {test_mae_cm_cal:.4f} cm")

    print("\nArquivos principais:")
    for name in (
        "configs/best_config.json",
        "configs/experiment_config.json",
        "metrics/window_level_metrics.csv",
        "metrics/ranking_final.csv",
        "predictions/etapa15_test_predictions.csv",
        "plots/etapa16_test_bland_altman.png",
    ):
        p = config.output_dir / name
        if p.is_file() or (name == "metrics/ranking_final.csv" and run_session_dir):
            actual = run_session_dir / "ranking_final.csv" if "ranking" in name and run_session_dir else p
            if actual.is_file():
                print(f"  • {actual}")

    if run_session_dir:
        print(f"Sessão de busca        : {run_session_dir}")

    print(f"Gráficos gerados       : {len(plot_paths)}")
    print(f"Tempo salvo em         : {timing_path}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline Deep Learning completo (Etapas 1–16).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_pipeline.py --device cpu --cpu-full-throttle --diagnostics --baselines \\
      --with-hyperparams-fast --refine-top-k 5 --stability-seeds --pruning \\
      --calibrate-from-loso --max-epochs 300 --patience 30

  python run_pipeline.py --device cpu --resume --with-hyperparams-fast --refine-top-k 5

  python run_pipeline.py --device cpu --diagnostics --with-hyperparams-fast --max-epochs 50
        """,
    )
    parser.add_argument("--skip-existing", action="store_true", help="retomar folds LOSO já concluídos")
    parser.add_argument("--resume", action="store_true", help="retomar busca de hiperparâmetros interrompida")
    parser.add_argument("--with-hyperparams", action="store_true", help="alias de --with-hyperparams-fast")
    parser.add_argument("--with-hyperparams-fast", action="store_true", help="busca progressiva (coarse + refine)")
    parser.add_argument("--all", action="store_true", help="grid legado completo (não recomendado)")
    parser.add_argument("--refine-top-k", type=int, default=0, metavar="N", help="refinar top-N da busca rápida")
    parser.add_argument("--stability-seeds", action="store_true", help="testar estabilidade com múltiplas sementes")
    parser.add_argument("--pruning", action="store_true", help="pruning moderado na busca")
    parser.add_argument("--pruning-warmup-epochs", type=int, default=30)
    parser.add_argument("--pruning-margin", type=float, default=1.50)
    parser.add_argument("--baselines", action="store_true", help="baselines obrigatórios no LOSO")
    parser.add_argument("--diagnostics", action="store_true", help="diagnósticos antes do treino")
    parser.add_argument("--calibrate-from-loso", action="store_true", help="calibração linear pós-modelo (sem vazamento)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-full-throttle", action="store_true", help="usar todos os cores CPU")
    parser.add_argument("--num-threads", type=int, default=None, metavar="N")
    parser.add_argument("--num-workers", type=int, default=0, metavar="N", help="workers dos DataLoaders")
    parser.add_argument("--max-epochs", type=int, default=None, help="épocas máximas por fold (padrão 300)")
    parser.add_argument("--patience", type=int, default=None, help="early stopping patience (padrão 30)")
    parser.add_argument("--min-delta", type=float, default=None, help="early stopping min_delta (padrão 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=None, help="weight decay Adam (padrão 1e-4)")
    parser.add_argument("--final-epochs", type=int, default=None, help="épocas do treino final (prioridade sobre --epochs)")
    parser.add_argument("--epochs", type=str, default="auto", help="'auto' (P75 LOSO) ou inteiro para treino final")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-quality-plots", action="store_true")
    parser.add_argument("--loso-plots", action="store_true")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 17), metavar="N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all and (args.with_hyperparams or args.with_hyperparams_fast):
        print("Aviso: --all substitui busca progressiva; --with-hyperparams-fast será ignorado.")

    if args.all:
        print("Aviso: --all bruto não é recomendado. Use busca progressiva (--with-hyperparams-fast).")

    hyperparams = _parse_hyperparams_mode(args)

    device_obj, runtime = setup_runtime(
        device=args.device,  # type: ignore[arg-type]
        cpu_full_throttle=args.cpu_full_throttle,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
    )
    if args.device == "cpu":
        print("Dispositivo selecionado: CPU forçado pelo usuário")

    train_cfg = TrainingConfig()
    if args.max_epochs is not None:
        train_cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        train_cfg.patience = args.patience
    if args.min_delta is not None:
        train_cfg.min_delta = args.min_delta
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    train_cfg.num_workers = runtime.num_workers

    paths = create_output_dirs(build_default_config(PROJECT_ROOT).output_dir)
    n_epochs = _parse_final_epochs(args, paths["metrics"])

    # Salvar config.json com runtime
    config_payload = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": runtime.to_dict(),
        "training": {
            "max_epochs": train_cfg.max_epochs,
            "patience": train_cfg.patience,
            "min_delta": train_cfg.min_delta,
            "weight_decay": train_cfg.weight_decay,
        },
        "flags": {
            "hyperparams": hyperparams,
            "refine_top_k": args.refine_top_k,
            "stability_seeds": args.stability_seeds,
            "pruning": args.pruning,
            "baselines": args.baselines,
            "diagnostics": args.diagnostics,
            "calibrate_from_loso": args.calibrate_from_loso,
            "resume": args.resume,
        },
    }
    (paths["configs"] / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    run_full_pipeline(
        hyperparams=hyperparams,
        skip_existing=args.skip_existing,
        resume=args.resume,
        verbose=args.verbose,
        n_epochs=n_epochs,
        quality_plots=not args.no_quality_plots,
        loso_plots=args.loso_plots,
        from_stage=args.from_stage,
        runtime=runtime,
        device_obj=device_obj,
        train_cfg=train_cfg,
        diagnostics=args.diagnostics,
        baselines=args.baselines,
        refine_top_k=args.refine_top_k,
        stability_seeds=args.stability_seeds,
        enable_pruning=args.pruning,
        pruning_warmup_epochs=args.pruning_warmup_epochs,
        pruning_margin=args.pruning_margin,
        calibrate_from_loso=args.calibrate_from_loso,
    )


if __name__ == "__main__":
    main()
