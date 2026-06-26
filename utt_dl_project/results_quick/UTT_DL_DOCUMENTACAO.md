# Documentação — Pipeline UTT Deep Learning Residual

Gerado automaticamente em `UTT_DL_DOCUMENTACAO.md`.

## 1. Objetivo do modelo

Predizer **ponto a ponto** a curva de deslocamento vertical externo do Vicon (`vicon_esternoZ_cm`) a partir de sinais do smartphone (IMU + features físicas corrigidas), usando uma **rede residual** que corrige escala, offset, fase e deformações locais sobre um baseline linear.

## 2. Estrutura dos dados de entrada

Cada arquivo `*_alinhado_ml.csv` contém:

- `Time`
- Acelerômetro: `accX_m_s2`, `accY_m_s2`, `accZ_m_s2`
- Giroscópio: `gyroX_rad_s`, `gyroY_rad_s`, `gyroZ_rad_s`
- `velocidade_corrigida_smart_m_s`
- `deslocamento_corrigido_smart_m`
- Target: `vicon_esternoZ_cm`

## 3. Unidades das variáveis

| Variável | Unidade original | Unidade usada no pipeline |
|----------|------------------|---------------------------|
| deslocamento smart | m | **cm** (`× 100`) |
| velocidade smart | m/s | **cm/s** (`× 100`) |
| Vicon | cm | cm (target) |

## 4. Conversão metros → centímetros

```python
smart_disp_cm = deslocamento_corrigido_smart_m * 100
smart_vel_cm_s = velocidade_corrigida_smart_m_s * 100
```

## 5. Split 70/30 por sujeito

- **27** sujeitos no total
- Treino: ['02', '03', '04', '06', '08', '09', '10', '13', '15', '17', '18', '21', '22', '23', '25', '26', '27', '28', '29']
- Validação: ['01', '11', '12', '14', '16', '19', '20', '24']
- Seed: 42
- Arquivo: `split_70_30_subjects.csv`

## 6. Por que não random split de janelas

Janelas do mesmo sujeito compartilham características biomecânicas e de sensor. Split aleatório de janelas causaria **vazamento** (treino e validação veriam o mesmo sujeito), inflando métricas artificialmente.

## 7. Baseline raw

Compara `smart_disp_cm` diretamente com Vicon, sem calibração.

## 8. Baseline linear

Ajustado **apenas nos sujeitos de treino**:

```
vicon ≈ a × smart_disp_cm + b
smart_calibrado_cm = a × smart_disp_cm + b
```

Coeficientes: a = 0.824316, b = 2.860554

## 9. Modelo residual

A rede prediz:

```
residuo = vicon_esternoZ_cm - smart_calibrado_cm
pred_dl_cm = smart_calibrado_cm + residuo_predito
```

Motivo: a dupla integral corrigida já traz forma física útil; a DL atua como calibradora temporal/não linear.

## 10. Arquitetura TCN residual

- Entrada: 10 canais × T amostras
- Blocos convolucionais 1D causais com dilatações `[1, 2, 4, 8, 16, 32, 64]`
- GroupNorm + GELU + Dropout
- Saída: resíduo por timestep (cm)
- Parâmetros: 273793

## 11. Janelamento temporal

- `window_size = 512`
- `stride = 128`
- Janelas criadas separadamente por split (treino/validação)

## 12. Reconstrução da curva inteira

Inferência com janela deslizante, overlap e **média ponderada Hann** nas bordas para suavizar costuras.

## 13. Loss composta (sobre curva final)

| Componente | Peso |
|------------|------|
| SmoothL1(pred_dl, vicon) | 0.50 |
| MSE(diff temporal) | 0.20 |
| Erro amplitude pico-a-pico | 0.15 |
| 1 − correlação | 0.15 |

## 14. Métricas calculadas

RMSE, MAE, Pearson r, R², erro de amplitude, erro percentual de amplitude, erro de pico, offset/média, lag por cross-correlation (amostras e segundos), melhora % DL vs linear.

## 15. Interpretação dos resultados

- **RMSE/MAE menores** → melhor aderência ponto a ponto
- **Pearson r próximo de 1** → forma temporal preservada
- **Lag ≈ 0** → boa sincronização temporal
- **Melhora vs linear** → evidência de que a DL corrige além da calibração afim

## 16. Limitações metodológicas

- Poucos sujeitos (~25); holdout 70/30 é exploratório
- Sem LOSO nesta versão (função futura)
- Baseline linear global (não por sujeito)
- Vicon usado **somente** como supervisão — nunca na correção de drift

## 17. Como rodar no Windows

```bash
cd utt_dl_project
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results --device auto
```

Teste rápido:

```bash
python run_utt_dl.py --data_dir ../Inputs_DP --out_dir ./results_quick --device auto --quick
```

## 18. AMD / DirectML / ROCm / CPU

- `--device auto`: detecta CUDA → ROCm → DirectML → MPS → CPU
- `--device directml`: GPU AMD no Windows (requer `torch-directml`, Python 3.11)
- `--device cpu`: força CPU
- Mixed precision apenas em CUDA/ROCm seguro

Device usado nesta execução: **cpu** (CPU (forçado))

## 19. Arquivos gerados

| Arquivo | Descrição |
|---------|-----------|
| `split_70_30_subjects.csv` | Sujeitos treino/validação |
| `metrics/metrics_by_file.csv` | Métricas por arquivo e método |
| `metrics/metrics_summary.csv` | Resumo agregado |
| `metrics/baseline_summary.csv` | Comparação baselines |
| `predictions/*_pred.csv` | Curvas completas preditas |
| `models/best_model.pt` | Melhor checkpoint |
| `plots/` | Gráficos |
| `training_time_log.csv` | Tempo por época |
| `training_time_summary.txt` | Resumo temporal |

## 20. Tempo e desempenho computacional

| Etapa | Tempo |
|-------|-------|
| Leitura dos dados | 0.2 s |
| Treinamento | 115.9 s |
| Avaliação/inferência | 1.6 s |
| Script total | 126.3 s |
| Melhor época | 16 |
| Melhor val_loss | 0.5705087631940842 |
| Melhor val_RMSE | 1.5904546082019806 cm |

### Resultados desta execução

| Método | RMSE médio (cm) |
|--------|-----------------|
| Smart raw | 3.2331 |
| Linear | 2.0101 |
| DL residual | 1.5340 |
| Melhora DL vs linear | 23.69% |
