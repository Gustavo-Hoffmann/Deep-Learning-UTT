# Pipeline UTT — Deep Learning Residual

Predição ponto a ponto do deslocamento vertical Vicon a partir de IMU + features físicas do smartphone (UTT).

## Estrutura

```
utt_dl_project/
  run_utt_dl.py          # ponto de entrada
  requirements.txt
  requirements-amd.txt   # opcional: GPU AMD DirectML
  README_RUN.md
  src/
    data.py              # leitura, split, baseline, janelas
    models.py            # TCN residual
    losses.py            # loss composta
    train.py             # treino + early stopping
    evaluate.py          # inferência completa + métricas
    plots.py             # gráficos
    documentation.py     # Markdown automático
    utils.py             # device, seed, timing
  results/               # criado na execução
```

## Dados

Coloque os CSVs `*_alinhado_ml.csv` em uma pasta (padrão: `../Inputs_DP` relativo ao projeto).

Colunas obrigatórias: `Time`, acelerômetro, giroscópio, `velocidade_corrigida_smart_m_s`, `deslocamento_corrigido_smart_m`, `vicon_esternoZ_cm`.

## Instalação

### CPU ou PyTorch genérico

```bash
cd utt_dl_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### GPU AMD no Windows (DirectML)

Requer **Python 3.11**:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-amd.txt
```

## Comandos

Execução normal:

```bash
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results --device auto
```

Teste rápido (20 épocas):

```bash
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results_quick --device auto --quick
```

Forçar CPU:

```bash
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results_cpu --device cpu
```

DirectML (se instalado):

```bash
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results_directml --device directml
```

## Saídas principais

| Arquivo | Conteúdo |
|---------|----------|
| `split_70_30_subjects.csv` | Sujeitos treino/validação |
| `models/best_model.pt` | Melhor checkpoint |
| `metrics/metrics_by_file.csv` | RMSE, MAE, r, lag, etc. |
| `metrics/baseline_summary.csv` | raw vs linear vs DL |
| `predictions/*_pred.csv` | Curvas completas |
| `plots/` | Gráficos por arquivo e resumo |
| `UTT_DL_DOCUMENTACAO.md` | Documentação gerada |
| `training_time_log.csv` | Tempo por época |

## Metodologia (resumo)

1. Split **70/30 por sujeito** (sem vazamento entre janelas)
2. Baseline linear `vicon ≈ a·smart_disp + b` ajustado só no treino
3. TCN prediz **resíduo** temporal: `pred = smart_calibrado + residuo_DL`
4. Loss composta sobre a curva final (forma + derivada + amplitude + correlação)
5. Inferência em sequência completa com overlap + janela Hann

## Device

`--device auto` detecta, nesta ordem: CUDA NVIDIA → ROCm → DirectML → MPS → CPU.

Não força CUDA nem altera seu ambiente AMD existente.
