#!/usr/bin/env python3
"""Monta o guia explicativo a partir do transcript (uso único)."""
import json
import re
from pathlib import Path

TRANSCRIPT = Path(
    "/Users/Rodacki/.cursor/projects/Users-Rodacki-Desktop-Hoffmann-UTT/agent-transcripts/"
    "1a285f07-19fc-4bed-9b02-8452e1c84bfd/1a285f07-19fc-4bed-9b02-8452e1c84bfd.jsonl"
)
OUT = Path("/Users/Rodacki/Desktop/Hoffmann/UTT/GUIA_EXPLICATIVO_PIPELINE.md")

SKIP_SECTIONS = {
    "3. Código Python/PyTorch",
    "3. Código Python",
    "3. Como rodar no futuro",
    "4. Comentário didático do código",
    "4. Saídas geradas ao concluir",
    "Como rodar no futuro",
    "Saídas geradas ao concluir",
}

def strip_code_blocks(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text

def remove_section(text: str, heading: str) -> str:
    pattern = rf"\n## {re.escape(heading)}[\s\S]*?(?=\n## |\n---\n\n## |\Z)"
    return re.sub(pattern, "\n", text, count=1)

REPLACEMENTS = [
    (r"use Path\(__file__\)\.parent", "use caminhos relativos ao projeto"),
    (r"sempre verifique get_device\(\)", "sempre confirme se está usando CPU ou GPU"),
    (r"inverse_transform_target\(\)", "reversão da normalização do alvo"),
    (r"\.fit\(\)", "ajuste do escalador"),
    (r"\.transform\(\)", "aplicação do escalador (sem reajuste)"),
    (r"model\.eval\(\)", "modo de avaliação do modelo"),
    (r"model\.train\(\)", "modo de treino do modelo"),
    (r"loss\.backward\(\)", "retropropagação"),
    (r"zero_grad\(\)", "zerar gradientes"),
    (r"shuffle=True", "embaralhamento ativado no treino"),
    (r"shuffle=False", "sem embaralhamento na validação"),
    (r"nn\.Conv1d", "convolução temporal"),
    (r"StandardScaler", "escalador padrão"),
    (r"ScalerBundle", "pacote de escaladores"),
    (r"DataLoader", "carregador de batches"),
    (r"__getitem__", "acesso a cada amostra"),
    (r"state_dict", "pesos salvos do modelo"),
    (r"--list", "modo prévia"),
    (r"--run", "modo execução"),
    (r"--epochs auto", "número automático de épocas"),
    (r"--quick-folds 3", "LOSO rápido com 3 sujeitos"),
    (r"--skip-existing", "retomar execução parcial"),
    (r"--source test", "fonte: teste final"),
    (r"--use-best-hparams", "usar melhores hiperparâmetros salvos"),
    (r"--epochs 14", "14 épocas manualmente"),
    (r"etapa\d+_[a-z_]+\.(csv|json|pt|joblib)", "arquivo de saída da etapa"),
    (r"outputs/pipeline_dl/plots/", "pasta de gráficos do experimento"),
    (r"loso_per_subject", "gráfico MAE por sujeito (LOSO)"),
    (r"O script aponta para lá[^\n]*\n", ""),
    (r"7\. \*\*Importar tudo dentro do loop de treino\*\*[^\n]*\n", ""),

def plain_language(text: str) -> str:
    for pat, repl in REPLACEMENTS:
        text = re.sub(pat, repl, text)
    return text

def clean_etapa_text(text: str) -> str:
    text = strip_code_blocks(text)
    text = re.sub(r"\n---\n\nQuando quiser seguir[\s\S]*?(?=\n\[REDACTED\]|\Z)", "\n", text)
    text = re.sub(r"\nQuando quiser[\s\S]*?(?=\n\[REDACTED\]|\Z)", "\n", text)
    text = re.sub(r"Quer que eu implemente[\s\S]*?(?=\n## |\Z)", "", text)
    text = re.sub(r"\[REDACTED\]", "", text)
    text = re.sub(r"\n## O que fica pronto após a Etapa \d+\n[\s\S]*?(?=\n## |\n---|\Z)", "\n", text)
    for sec in SKIP_SECTIONS:
        text = remove_section(text, sec)
    text = re.sub(r"Script criado em pipeline/[^\n]+\.\s*", "", text)
    text = re.sub(r"Script pronto em pipeline/[^\n]+\.\s*", "", text)
    text = re.sub(r"Módulo criado em pipeline/[^\n]+\.\s*", "", text)
    text = re.sub(r"\*\*Nada foi treinado agora\*\*[^\n]*\n", "", text)
    text = re.sub(r"só validei com --list\.?\s*", "", text)
    text = re.sub(r"Não executei a geração completa de gráficos[^\n]*\n", "", text)
    text = re.sub(r"### Comandos[\s\S]*?(?=\n### |\n## |\n---|\Z)", "", text)
    text = re.sub(r"## Como rodar[\s\S]*?(?=\n## |\n---|\Z)", "", text)
    text = re.sub(r"\n---\n(?:\n---\n)+", "\n---\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return plain_language(text.strip())

def load_etapas() -> dict[int, str]:
    found: dict[int, str] = {}
    with open(TRANSCRIPT) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("role") != "assistant":
                continue
            for part in obj.get("message", {}).get("content", []):
                if part.get("type") != "text":
                    continue
                text = part["text"]
                for pat in (r"^# Etapa (\d+)", r"^## Etapa (\d+)"):
                    m = re.match(pat, text)
                    if m:
                        n = int(m.group(1))
                        t = text.replace("## Etapa", "# Etapa", 1) if text.startswith("##") else text
                        if n not in found or len(t) > len(found[n]):
                            found[n] = t
    return found

ETAPA_1_EXTRA = """
## 3. O que fica preparado nesta etapa

- Semente aleatória fixa, para repetir splits e treinos.
- Escolha automática do dispositivo de computação (CPU, GPU NVIDIA ou MPS no Mac).
- Parâmetros centrais do experimento: frequência, tamanho da janela, stride, tipo de modelo, seed, pasta de dados e pasta de saídas.
- Estrutura de pastas para checkpoints, métricas, gráficos, splits, scalers e configs.
- Registro em JSON da configuração e versões das bibliotecas (útil para reprodutibilidade e publicação).
"""

ETAPA_13 = """
# Etapa 13 — LOSO completo nos 70% de desenvolvimento

## 1. O conceito, de forma simples

A Etapa 12 treinou **um único fold** (validação = um sujeito dev). A Etapa 13 repete esse procedimento para **todos os 19 sujeitos de desenvolvimento**, um de cada vez.

Leave-One-Subject-Out (LOSO) significa: em cada rodada, **18 sujeitos treinam** e **1 sujeito fica de fora para validação**. Esse sujeito de validação nunca entra no ajuste do escalador nem no treino daquele fold.

Para cada fold, o pipeline refaz do zero:
- normalização (escalador ajustado só nos 18 de treino);
- criação de janelas;
- treino com early stopping;
- métricas no sujeito de validação.

Ao final, você tem **19 métricas** (uma por sujeito dev) e pode calcular **média ± desvio-padrão** — estimativa honesta de generalização para um **novo indivíduo** antes de tocar o teste final.

O grupo de **teste final (30%) permanece intocado**.

---

## 2. Por que isso importa no seu problema

| Objetivo | O que o LOSO responde |
|---|---|
| Generalização por pessoa | O modelo funciona em corpos que não viu no treino daquele fold? |
| Variabilidade entre sujeitos | Alguns indivíduos são sistematicamente mais difíceis? |
| Baseline antes do teste | Qual MAE esperar internamente antes da avaliação final? |
| Escolha de hiperparâmetros (Etapa 14) | Comparar configs com o mesmo protocolo rigoroso |

LOSO por sujeito é o padrão ouro quando cada pessoa traz postura, ritmo, posição do telefone e biomecânica próprios — exatamente o seu cenário IMU → Vicon.

---

## 3. Resultado obtido (TCN baseline, config padrão)

Execução completa: **19 folds**, ~19 minutos em CPU.

| Métrica agregada | Valor (escala normalizada do alvo) |
|---|---|
| MAE média | 0,205 ± 0,152 |
| RMSE média | 0,253 ± 0,172 |
| Bias médio | −0,020 ± 0,199 |

Piores sujeitos (MAE): **15** (0,634), **12** (0,541), **03** (0,354).

Melhores sujeitos (MAE): **20** (0,085), **19** e **08** (~0,105).

Interpretação: a maioria generaliza bem, mas **outliers por sujeito** existem — útil para inspeção clínica e de qualidade do sinal (Etapa 16).

Sobre R² por fold: com ~31–32 janelas de validação por sujeito, R² individual pode ser instável ou negativo mesmo com MAE baixo. Para interpretação, priorize **MAE e RMSE agregados**.

Mediana da melhor época entre folds: **6** (usada como sugestão na Etapa 15 com épocas automáticas).

---

## 4. Principais erros a evitar

1. **Incluir sujeitos de teste final** — LOSO usa só os 19 dev.
2. **Um único escalador para todos os folds** — cada fold tem seu próprio ajuste de normalização.
3. **Escolher hiperparâmetros olhando o teste** — teste só na Etapa 15.
4. **Confundir LOSO com treino final** — LOSO mede; treino final usa 100% dev.
5. **Ignorar sujeitos outliers** — MAE alto em um fold pode indicar problema de sinal, não só do modelo.
6. **Interromper e perder progresso** — a execução pode ser retomada a partir dos folds já concluídos.
"""

ETAPA_16 = """
# Etapa 16 — Visualização dos resultados

## 1. O conceito, de forma simples

Depois de obter métricas numéricas (LOSO na Etapa 13 e/ou teste final na Etapa 15), a visualização traduz os números em **figuras interpretáveis** para você, orientadores e revisores.

Duas fontes possíveis:

- **Teste final (Etapa 15):** predições janela a janela nos 8 sujeitos intocados — visão completa de generalização externa.
- **LOSO dev (Etapa 13):** MAE por sujeito nos 19 folds — visão de generalização interna antes do teste.

## 2. Gráficos produzidos

**True vs predito:** cada ponto é uma janela. Proximidade à diagonal ideal (predição = realidade) indica boa predição. Dispersão lateral sugere erro aleatório; curvatura sistemática sugere viés.

**Bland-Altman:** compara a média entre real e predito com a diferença (pred − real). Revela bias constante e se o erro cresce com a amplitude (heterocedasticidade).

**Resíduos:** distribuição de (pred − real). Ideal: centrada em zero e simétrica. Caudas pesadas indicam erros grandes ocasionais; deslocamento indica bias.

**Erro por sujeito:** MAE por indivíduo de teste ou de cada fold LOSO. Identifica quem generaliza mal — outliers merecem inspeção clínica e de qualidade do sinal.

**Erro por faixa de amplitude:** MAE médio em movimentos pequenos vs grandes — relevante para validade clínica.

**Timeline:** série temporal de amplitudes preditas vs reais, por sujeito. Mostra se o modelo acompanha tendências ou apenas a média.

**LOSO por sujeito:** resumo da generalização interna antes do teste final.

## 3. Estado e ordem recomendada

- LOSO (Etapa 13): **já executado** — gráficos de MAE por sujeito dev já podem ser gerados.
- Teste final (Etapa 15): **pendente** — gráficos completos dependem das predições do teste.

Ordem sugerida: Etapa 14 (opcional) → Etapa 15 → Etapa 16 com teste final; gráficos LOSO podem ser feitos a qualquer momento após a Etapa 13.

## 4. Principais erros a evitar

1. **Interpretar gráficos do teste antes de rodar a Etapa 15** — predições de teste ainda não existem.
2. **Concluir validade clínica só pelo gráfico** — complemente com MAE em centímetros e discussão qualitativa (Etapa 18, futura).
3. **Ignorar sujeitos outliers nos gráficos** — podem indicar problema de aquisição, não só do modelo.
4. **Misturar resultados de configs diferentes** — use sempre os artefatos da mesma execução (mesma config, mesmos hiperparâmetros).
"""

ETAPA_16_EXTRA = """
---

# Visão geral do pipeline (16 etapas)

O fluxo metodológico fixo é:

1. Preparar ambiente e configuração
2. Definir contrato dos dados
3. Carregar arquivos por sujeito
4. Conferir qualidade e alinhamento temporal
5. Split externo 70/30 por sujeito
6. Normalizar sem vazamento (por fold)
7. Criar janelas temporais e alvo (amplitude)
8. Montar Dataset e DataLoader PyTorch
9. Modelo baseline CNN 1D
10. Modelo principal TCN
11. Loss, otimizador e métricas
12. Treinar um fold (prova de conceito)
13. LOSO completo nos 70% dev
14. Busca de hiperparâmetros (LOSO dev)
15. Treino final + teste intocado (30%)
16. Visualização e interpretação dos resultados

Regras que atravessam tudo: split por sujeito, teste final congelado até a Etapa 15, normalização e janelas refeitas a cada fold, métricas clínicas em centímetros após reversão do escalador do alvo.

---

# Prévia das etapas 17–20 (não implementadas)

Estas etapas foram **planejadas** como extensão natural do pipeline, mas **ainda não existem** como scripts. O pipeline oficial termina na Etapa 16.

## Etapa 17 — Comparação com baselines

**Ideia:** contextualizar o desempenho da rede profunda contra métodos mais simples, respondendo: *“vale a pena usar deep learning?”*

Baselines típicos para IMU → deslocamento/amplitude:

- **Dupla integração** da aceleração (com ou sem remoção de gravidade) — mostra por que a abordagem física pura falha (drift, orientação).
- **Média ou mediana** da amplitude no grupo de treino — baseline estatístico mínimo.
- **Machine learning clássico** (Random Forest, XGBoost, regressor linear) com features hand-crafted por janela (RMS, picos, energia espectral dos 6 canais).

**Protocolo:** mesmos splits, mesmas janelas, mesmas métricas (MAE/RMSE em cm). Comparar no LOSO dev e, depois, no teste final.

## Etapa 18 — Análise de erro e validade clínica

**Ideia:** ir além do MAE agregado e discutir **utilidade clínica**.

Análises possíveis:

- Erro por faixa de amplitude, velocidade ou tipo de movimento.
- Limites de concordância (Bland-Altman em cm).
- Erro máximo tolerável (MCID ou limiar definido com orientação clínica).
- Correlação entre erro e duração da gravação, posição do telefone, etc.
- Discussão de sujeitos outliers (ex.: 12, 15 no LOSO).

## Etapa 19 — Salvamento completo e reprodutibilidade

**Ideia:** empacotar tudo para repetir o experimento meses depois ou compartilhar com revisores.

Conteúdo:

- Snapshot final de configuração, seed, versões de bibliotecas.
- Lista de sujeitos dev/teste, hiperparâmetros vencedores.
- Pesos do modelo final, escalador, predições, métricas, gráficos.
- README de execução passo a passo e hash ou checksum dos dados de entrada.

## Etapa 20 — Escrita metodológica (artigo/tese)

**Ideia:** transformar o pipeline em **texto científico** — Métodos, Resultados, Discussão.

Estrutura sugerida:

- Descrição dos participantes e aquisição (IMU + Vicon alinhados).
- Pré-processamento, janelamento, normalização sem vazamento.
- Arquiteturas (CNN 1D, TCN), treino, LOSO, seleção de hiperparâmetros.
- Avaliação externa no 30% intocado.
- Limitações (N pequeno, CPU, amplitude escalar vs curva).
- Figuras-chave da Etapa 16 e tabelas de métricas.

---

# Como adicionar as etapas 17–20 depois

## Princípio de encaixe

Cada nova etapa deve ser um módulo independente, reutilizando a lógica das etapas anteriores **sem alterar** o comportamento das Etapas 1–16. O padrão já usado:

- Reaproveitar configuração, carregamento, split e normalização já definidos.
- Salvar saídas na pasta organizada do experimento (métricas, gráficos, configs, predições).
- Oferecer modo de prévia e modo de execução, quando fizer sentido.
- Deixar explícito o que a etapa **não** faz (por exemplo: não usar teste final).

## Ordem sugerida de implementação

1. **Etapa 17** — baselines; depende de Etapas 5–7 e 11; pode rodar em paralelo ao modelo DL usando os mesmos folds.
2. **Etapa 18** — análise clínica; depende de predições da Etapa 15 (e opcionalmente LOSO da 13).
3. **Etapa 19** — empacotamento; depende de 13–16 concluídas (idealmente 15 para teste final).
4. **Etapa 20** — redação; depende de resultados numéricos e figuras das etapas anteriores (pode ser um documento Markdown/LaTeX separado, sem execução automática).

## Convenções de nomenclatura

- Novo módulo numerado sequencialmente (Etapa 17, 18, …) na pasta do pipeline.
- Prefixo de artefatos: etapa17_ (ou número correspondente) nos arquivos de saída.
- Atualizar este guia explicativo com a seção da etapa quando ela for implementada.

## O que não mudar

- Split 70/30 por sujeito (seed 42) permanece fixo.
- Teste final continua intocado até uma única avaliação final.
- Baselines e novos modelos devem respeitar o mesmo protocolo de validação — senão a comparação deixa de ser justa.

---

*Guia gerado a partir das explicações do desenvolvimento do pipeline UTT (IMU smartphone → amplitude Vicon). Pipeline implementado: Etapas 1–16.*
"""

INTRO = """# Guia explicativo — Pipeline Deep Learning IMU → Vicon (UTT)

Este documento reúne, em linguagem didática, **o que cada etapa do pipeline faz e por quê**. Não é manual de código: é o raciocínio metodológico por trás das decisões.

## Contexto do problema

Você tem dados **temporalmente alinhados**: em cada linha, seis canais de IMU de smartphone (aceleração e giroscópio) e a referência **Vicon** (deslocamento/amplitude precisa de um ponto corporal). Cada **arquivo corresponde a um sujeito**.

O objetivo é **predizer a amplitude do movimento Vicon** a partir do padrão temporal do IMU em janelas curtas (inicialmente: amplitude escalar = máximo − mínimo do Vicon na janela).

A integração analítica da aceleração falha na prática (drift, ruído, orientação). Por isso o pipeline aprende uma **relação empírica** com redes temporais (CNN 1D e TCN), respeitando validação rigorosa por sujeito.

## Regras metodológicas fixas

- **Split externo 70/30 por sujeito:** 70% desenvolvimento, 30% teste final intocado até a avaliação definitiva.
- **LOSO nos 70%:** validação interna deixando um sujeito de fora por vez.
- **Sem vazamento:** normalização, hiperparâmetros e janelas ajustados só com dados permitidos em cada fase.
- **Alvo inicial:** amplitude escalar por janela.
- **Modelos:** CNN 1D (baseline) e TCN (principal).

## Dados do projeto

- Pasta de entrada: Input_ML/ — 27 arquivos CSV, sujeitos 01–29 (faltam 05 e 07).
- ~53 mil amostras no total, **60 Hz**, gravações típicas de ~22–37 s por sujeito.
- Colunas reais mapeadas para nomes canônicos (time, acc_x … gyro_z, vicon).
- Nenhum valor ausente nas colunas obrigatórias.

---
"""

def main():
    etapas = load_etapas()
    etapas[13] = ETAPA_13.strip()
    if 1 in etapas:
        etapas[1] = etapas[1].replace(
            "No seu projeto, os dados já existem em Input_ML/",
            ETAPA_1_EXTRA.strip() + "\n\nNo seu projeto, os dados já existem em Input_ML/",
        )
    if 16 in etapas:
        etapas[16] = etapas[16] + "\n\n" + ETAPA_16_EXTRA.strip()

    parts = [INTRO.strip()]
    for n in range(1, 17):
        if n not in etapas:
            continue
        body = clean_etapa_text(etapas[n])
        parts.append(body)
        parts.append("\n---\n")

    parts.append(FUTURO.strip())
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Escrito: {OUT} ({OUT.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
