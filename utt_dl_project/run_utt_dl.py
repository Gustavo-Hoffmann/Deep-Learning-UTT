#!/usr/bin/env python3
"""
Pipeline UTT — Deep Learning residual para predição de deslocamento Vicon.

Uso:
  python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results --device auto
  python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results_quick --device auto --quick
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

# Garante import do pacote src quando executado diretamente
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.data import (  # noqa: E402
    WindowDataset,
    fit_linear_baseline,
    fit_scalers,
    load_all_files,
    print_load_report,
    save_split_csv,
    split_by_subject,
)
from src.documentation import generate_documentation  # noqa: E402
from src.evaluate import (  # noqa: E402
    add_improvement_column,
    evaluate_all_files,
    metrics_to_dataframe,
    summarize_metrics,
)
from src.models import ResidualTCN, count_parameters  # noqa: E402
from src.plots import generate_all_plots  # noqa: E402
from src.train import TrainConfig, build_loaders, train_model  # noqa: E402
from src.utils import (  # noqa: E402
    RunTimer,
    create_output_dirs,
    default_data_dir,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline UTT DL residual — Vicon a partir de smartphone")
    p.add_argument("--data_dir", type=Path, default=None, help="Pasta com *_alinhado_ml.csv")
    p.add_argument("--out_dir", type=Path, default=Path("results"), help="Pasta de saída")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "rocm", "directml", "mps"), default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_size", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quick", action="store_true", help="Modo rápido: 20 épocas, menos plots")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_dir = (args.data_dir or default_data_dir()).resolve()
    out_dir = args.out_dir.resolve()
    dirs = create_output_dirs(out_dir)

    device, device_desc, backend = resolve_device(args.device)  # type: ignore[arg-type]
    print("=" * 70)
    print("Pipeline UTT — Deep Learning Residual")
    print("=" * 70)
    print(f"Data dir   : {data_dir}")
    print(f"Out dir    : {out_dir}")
    print(f"Device     : {device} ({device_desc})")
    print(f"Modo       : {'quick' if args.quick else 'normal'}")
    print()

    script_timer = RunTimer()
    data_timer = RunTimer()

    # --- 1. Carregar dados ---
    report = load_all_files(data_dir)
    print_load_report(report)
    if report.n_files == 0:
        print("Nenhum arquivo válido. Abortando.")
        return 1
    data_load_s = data_timer.elapsed()

    # --- 2. Split por sujeito ---
    split = split_by_subject(report.files_ok, seed=args.seed)
    save_split_csv(split, out_dir / "split_70_30_subjects.csv")
    print(f"\nSplit 70/30 por sujeito:")
    print(f"  Treino ({len(split.train_subjects)}): {split.train_subjects}")
    print(f"  Val    ({len(split.val_subjects)}): {split.val_subjects}")

    # --- 3. Baseline linear (só treino) ---
    baseline = fit_linear_baseline(split.train_files)
    print(f"\nBaseline linear (treino): vicon ~ {baseline.slope:.6f} * smart_disp + {baseline.intercept:.6f}")

    # --- 4. Scalers (só treino) ---
    scalers = fit_scalers(
        split.train_files,
        baseline,
        window_size=args.window_size,
        stride=args.stride,
        scale_residual=True,
    )

    # Salvar artefatos
    with open(out_dir / "models" / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open(out_dir / "models" / "baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)

    # --- 5. Datasets e modelo ---
    train_ds = WindowDataset(
        split.train_files,
        baseline,
        scalers,
        window_size=args.window_size,
        stride=args.stride,
    )
    val_ds = WindowDataset(
        split.val_files,
        baseline,
        scalers,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"\nJanelas treino: {len(train_ds)} | validação: {len(val_ds)}")

    if len(train_ds) == 0:
        print("Nenhuma janela de treino. Reduza window_size ou verifique dados.")
        return 1

    model = ResidualTCN(
        n_input_channels=10,
        window_size=args.window_size,
        num_channels=96 if not args.quick else 64,
        kernel_size=5,
        dropout=0.15,
        dml_safe=(backend == "directml"),
    )
    n_params, _ = count_parameters(model)
    dml_note = " [modo DirectML: ReLU, sem GroupNorm/weight_norm]" if backend == "directml" else ""
    print(f"Modelo TCN: {n_params:,} parametros | receptive field = {model.receptive_field}{dml_note}")

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        quick=args.quick,
    )
    train_loader, val_loader = build_loaders(train_ds, val_ds, train_cfg)

    # --- 6. Treinar ---
    print("\n--- Treinamento ---")
    train_result = train_model(
        model,
        train_loader,
        val_loader,
        scalers,
        device=device,
        backend=backend,
        out_dir=out_dir,
        cfg=train_cfg,
    )

    # --- 7. Avaliação em sequências completas (validação) ---
    eval_timer = RunTimer()
    print("\n--- Avaliação (sequências completas de validação) ---")
    metrics, pred_dfs = evaluate_all_files(
        model,
        split.val_files,
        baseline,
        scalers,
        device=device,
        split_label="validation",
        predictions_dir=dirs["predictions"],
        window_size=args.window_size,
        stride=args.stride,
    )
    eval_s = eval_timer.elapsed()

    metrics_df = add_improvement_column(metrics_to_dataframe(metrics))
    summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(dirs["metrics"] / "metrics_by_file.csv", index=False)
    summary_df.to_csv(dirs["metrics"] / "metrics_summary.csv", index=False)

    baseline_summary = summary_df[["method", "rmse_cm_mean", "mae_cm_mean", "pearson_r_mean", "r2_mean"]].copy()
    if "improvement_vs_linear_pct" in summary_df.columns:
        dl_row = summary_df.loc[summary_df["method"] == "dl_residual"]
        if not dl_row.empty:
            baseline_summary.loc[baseline_summary["method"] == "dl_residual", "improvement_vs_linear_pct"] = (
                dl_row["improvement_vs_linear_pct"].iloc[0]
            )
    baseline_summary.to_csv(dirs["metrics"] / "baseline_summary.csv", index=False)

    # --- 8. Plots ---
    print("\n--- Gerando plots ---")
    generate_all_plots(pred_dfs, metrics_df, dirs["plots"], quick=args.quick)

    # --- 9. Documentação ---
    def _rmse(method: str) -> float:
        row = summary_df.loc[summary_df["method"] == method, "rmse_cm_mean"]
        return float(row.iloc[0]) if len(row) else float("nan")

    rmse_raw = _rmse("raw_smart")
    rmse_lin = _rmse("linear_baseline")
    rmse_dl = _rmse("dl_residual")
    improvement = 100.0 * (rmse_lin - rmse_dl) / max(rmse_lin, 1e-8)

    total_s = script_timer.elapsed()
    doc_ctx = {
        "n_subjects": report.n_subjects,
        "train_subjects": split.train_subjects,
        "val_subjects": split.val_subjects,
        "seed": args.seed,
        "baseline_slope": baseline.slope,
        "baseline_intercept": baseline.intercept,
        "n_params": n_params,
        "window_size": args.window_size,
        "stride": args.stride,
        "device": str(device),
        "device_desc": device_desc,
        "data_load_s": data_load_s,
        "train_s": train_result.train_seconds,
        "eval_s": eval_s,
        "total_s": total_s,
        "best_epoch": train_result.best_epoch,
        "best_val_loss": train_result.best_val_loss,
        "best_val_rmse": train_result.best_val_rmse_cm,
        "rmse_raw": rmse_raw,
        "rmse_linear": rmse_lin,
        "rmse_dl": rmse_dl,
        "improvement_pct": improvement,
    }
    generate_documentation(out_dir / "UTT_DL_DOCUMENTACAO.md", context=doc_ctx)

    # --- Resumo final ---
    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(f"Arquivos lidos              : {report.n_files}")
    print(f"Sujeitos                    : {report.n_subjects}")
    print(f"Sujeitos treino             : {split.train_subjects}")
    print(f"Sujeitos validação          : {split.val_subjects}")
    print(f"Modo validação              : holdout 70/30 por sujeito")
    print(f"Device                      : {device} ({device_desc})")
    print(f"RMSE médio smart raw        : {rmse_raw:.4f} cm")
    print(f"RMSE médio baseline linear  : {rmse_lin:.4f} cm")
    print(f"RMSE médio DL residual      : {rmse_dl:.4f} cm")
    print(f"Melhora DL vs linear        : {improvement:.2f}%")
    print(f"Melhor época                : {train_result.best_epoch}")
    print(f"Tempo leitura dados         : {data_load_s:.1f} s")
    print(f"Tempo treino                : {train_result.train_seconds:.1f} s")
    print(f"Tempo avaliação             : {eval_s:.1f} s")
    print(f"Tempo total script          : {total_s:.1f} s")
    print(f"Resultados                  : {out_dir}")
    print(f"Documentação                : {out_dir / 'UTT_DL_DOCUMENTACAO.md'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
