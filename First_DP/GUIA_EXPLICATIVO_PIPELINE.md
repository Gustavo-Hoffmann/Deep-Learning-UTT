# Guia explicativo — Pipeline Deep Learning IMU → Vicon (UTT)

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

# Etapa 1 — Preparação inicial do ambiente

## 1. O conceito, de forma simples

Antes de tocar nos dados ou treinar qualquer modelo, você monta a **base do laboratório**: importa as ferramentas, fixa a **semente aleatória** (para repetir resultados), escolhe onde o PyTorch vai calcular (CPU, GPU NVIDIA ou MPS no Mac), define **parâmetros fixos do experimento** e cria **pastas organizadas** para salvar resultados.

É como preparar bancada, instrumentos e caderno de protocolo antes de uma medição — nada de análise ainda, só garantir que tudo está no lugar certo.

---

## 2. Por que isso importa no seu problema

No seu caso (IMU do smartphone → deslocamento Vicon), o pipeline terá muitas decisões encadeadas: split 70/30 por sujeito, LOSO, normalização sem vazamento, janelas temporais, CNN/TCN, métricas clínicas.

Sem essa etapa inicial, você corre risco de:

- resultados **irreproduzíveis** (cada execução dá um número diferente);
- treinar na **GPU errada** ou na CPU sem perceber;
- perder **configuração** (qual janela? qual seed? quais sujeitos no teste?);
- misturar arquivos de **vários experimentos** na mesma pasta.

A Etapa 1 **não carrega dados** de propósito: primeiro você define *onde* e *como* o experimento vai rodar; na Etapa 3 você lê os arquivos.

---

## 3. O que fica preparado nesta etapa

- Semente aleatória fixa, para repetir splits e treinos.
- Escolha automática do dispositivo de computação (CPU, GPU NVIDIA ou MPS no Mac).
- Parâmetros centrais do experimento: frequência, tamanho da janela, stride, tipo de modelo, seed, pasta de dados e pasta de saídas.
- Estrutura de pastas para checkpoints, métricas, gráficos, splits, escaladores e configs.
- Registro em JSON da configuração e versões das bibliotecas (útil para reprodutibilidade e publicação).

No seu projeto, os dados já existem em Input_ML/ (ex.: 02_alinhado_ml.csv), mas **não são abertos** nesta etapa.

---

## 4. Principais erros a evitar

1. **Pular a semente** — resultados diferentes a cada execução, impossível comparar modelos.
2. **Hardcodar caminhos absolutos** — use caminhos relativos ao projeto para funcionar em qualquer máquina.
3. **Carregar dados nesta etapa** — qualquer olhada nos dados antes do split 70/30 aumenta risco de vazamento inconsciente.
4. **Misturar saídas de experimentos** — sempre use subpastas com timestamp ou nome do experimento.
5. **Assumir GPU disponível** — sempre confirme se está usando CPU ou GPU; no Mac pode ser MPS ou CPU.
6. **Não salvar a configuração** — sem registro da config, você não sabe qual janela ou seed foi usada meses depois.
7. **Confundir setup com treino** — esta etapa não cria janelas, não normaliza, não faz split.

---


# Etapa 2 — Estrutura esperada dos arquivos

## 1. O conceito, de forma simples

Antes de ler ou treinar qualquer coisa, você define o **contrato dos dados**: o que cada arquivo deve conter, como identificar o sujeito e o que o modelo vai prever.

No seu caso:

- **1 arquivo = 1 sujeito**
- **8 colunas obrigatórias** (tempo + 6 canais IMU + Vicon)
- **subject_id** vem do nome do arquivo (ex.: 02_alinhado_ml.csv → "02")
- **dois modos possíveis de alvo**: amplitude escalar ou curva temporal do Vicon

É como conferir o formulário de coleta antes de abrir as planilhas: você sabe o que esperar e detecta problemas cedo.

---

## 2. Por que isso importa no seu problema

Sem esse contrato explícito, erros comuns aparecem tarde demais:

| Risco | Consequência |
|---|---|
| Coluna com nome diferente (Time vs time) | Leitura falha ou usa coluna errada |
| Dois sujeitos com o mesmo subject_id | Vazamento no split 70/30 e no LOSO |
| Confundir amplitude com curva | Arquitetura e loss incompatíveis |
| Misturar janelas de sujeitos diferentes | Generalização falsa |

Seus arquivos reais em Input_ML/ usam nomes como Time, accX_m_s2, vicon_esternoZ_cm. O pipeline trabalha internamente com nomes canônicos (time, acc_x, vicon) e faz o **mapeamento** automaticamente.

Foram encontrados **27 arquivos** (sujeitos 01 a 29, com alguns números ausentes).

---

---

---

## 5. Principais erros a evitar

1. **Assumir nomes de colunas fixos** — sempre use mapeamento/aliases; seus arquivos usam Time, não time.
2. **Usar índice da linha como sujeito** — o sujeito vem do **arquivo**, não da linha.
3. **subject_id duplicado** — dois arquivos gerando o mesmo id quebram o split 70/30 e o LOSO.
4. **Validar colunas só no primeiro arquivo** — cada arquivo deve ser checado na Etapa 3.
5. **Confundir time com feature** — tempo organiza a série, mas **não entra** nas 6 entradas do modelo.
6. **Começar predizendo a curva inteira** — mais difícil; amplitude escalar é o caminho mais sólido para validar o pipeline.
7. **Carregar todos os dados nesta etapa** — aqui só definimos o contrato; leitura completa é Etapa 3.
8. **Ignorar unidades** — seus dados estão em m/s², rad/s e cm (Vicon); documente isso para interpretar métricas depois.

---


---


# Etapa 3 — Leitura dos arquivos por sujeito

## 1. O conceito, de forma simples

Depois de definir o contrato (Etapa 2), você **abre cada arquivo** e organiza os dados em uma estrutura clara: **um sujeito → um DataFrame**.

Para cada arquivo, o pipeline:
1. Lê o conteúdo (CSV ou XLSX)
2. Verifica se as colunas obrigatórias existem
3. Renomeia para o padrão canônico (time, acc_x, …, vicon)
4. Adiciona subject_id e source_file
5. Conta linhas e valores ausentes

Nada de split, normalização ou janelas ainda — só **carregar e inspecionar**.

---

## 2. Por que isso importa no seu problema

Com 27 arquivos e ~53 mil amostras no total, erros de leitura passariam despercebidos até o treino:

| Checagem | O que evita |
|---|---|
| Colunas obrigatórias | Treinar com feature faltando |
| subject_id único por arquivo | Misturar sujeitos no split 70/30 |
| Contagem de linhas | Sujeito com gravação truncada |
| Valores ausentes | NaN propagando para loss e métricas |
| Ordenação por time | Janelas temporais incoerentes |

A estrutura LoadedDataset guarda cada sujeito separado — base para split por sujeito (Etapa 5) e LOSO (Etapa 13).

---

---

---

## 5. Resultado na sua base real

| Métrica | Valor |
|---|---|
| Arquivos lidos | 27 |
| Sujeitos | 27 (IDs 01–29, faltam 05 e 07) |
| Amostras totais | 53 226 |
| Amostras/sujeito | min=1 325, mediana=1 982, max=2 198 |
| Duração típica | ~33 s por sujeito (~60 Hz) |
| Valores ausentes | **nenhum** nas 8 colunas obrigatórias |
| Sujeito menor | 26 — 1 325 amostras (~22 s) |

Colunas finais em cada DataFrame:

---

## 6. Principais erros a evitar

1. **Concatenar tudo e esquecer subject_id** — sem identificador, o split por sujeito fica impossível.
2. **Não validar cada arquivo** — um CSV corrompido pode quebrar o treino horas depois.
3. **Assumir mesma quantidade de linhas** — seus sujeitos variam de 1 325 a 2 198 amostras.
4. **Ignorar subject_id duplicado** — dois arquivos com prefixo 02 colapsariam num único sujeito.
5. **Normalizar nesta etapa** — normalização vem na Etapa 6, dentro de cada fold.
6. **Fazer split agora** — primeiro carregar tudo; separar grupos na Etapa 5.
7. **Usar índice da linha como tempo** — use a coluna time; ela define a ordem temporal real.
8. **Descartar metadados** — source_file ajuda a rastrear problemas até a publicação.

---


---


# Etapa 4 — Conferência pós-alinhamento e qualidade dos sinais

## 1. O conceito, de forma simples

Você já carregou os dados (Etapa 3) e assume que **cada linha** traz IMU e Vicon no **mesmo instante**. Agora confere se essa premissa se sustenta:

- o tempo avança de forma ordenada?
- a taxa de amostragem é estável (~60 Hz)?
- existem buracos no tempo?
- IMU e Vicon têm o mesmo número de amostras?
- os sinais parecem coerentes visualmente?

É uma **vistoria de qualidade** antes de dividir sujeitos e treinar — sem filtros pesados, sem normalização, sem split.

---

## 2. Por que isso importa no seu problema

Modelos temporais (CNN 1D, TCN) assumem que cada janela é uma sequência **regular e contínua**. Problemas silenciosos na base contaminam tudo depois:

| Problema | Efeito no modelo |
|---|---|
| Tempo não crescente | Janelas com ordem errada |
| Buracos temporais | Janela de "2 s" cobre mais tempo real |
| Frequência instável | Tamanho de janela em amostras perde sentido físico |
| IMU e Vicon desalinhados | O modelo aprende relação espúria |
| Sinal com artefato óbvio | Erro alto em sujeitos específicos |

Esta etapa inspeciona **todos os sujeitos igualmente**. Ainda não existe grupo de teste — portanto não há risco de vazamento por olhar dados de teste.

---

---

---

## 5. Resultado na sua base real

| Métrica | Resultado |
|---|---|
| Sujeitos analisados | 27 |
| Sem alertas | **27 / 27** |
| Frequência estimada | **60,00 Hz** em todos (dt = 1/60 ≈ 0,0167 s) |
| Tempo crescente | Sim, em todos |
| Buracos temporais | Nenhum |
| IMU = Vicon (contagem) | Sim, em todos |
| Duração | 22–37 s por sujeito (sujeito 26 é o mais curto) |

Plots salvos em:
- pasta de gráficos do experimentoetapa04_sinais_sujeito_01.png
- pasta de gráficos do experimentoetapa04_sinais_sujeito_16.png
- pasta de gráficos do experimentoetapa04_sinais_sujeito_29.png

**Conclusão:** os dados alinhados estão consistentes para avançar. A frequência de 60 Hz confirma o placeholder sampling_hz=60.0 da config — na Etapa 7, janela de 2 s = **120 amostras**.

---

## 6. Principais erros a evitar

1. **Aplicar filtro passa-banda agora** — pode mascarar problemas de alinhamento; filtrar só com justificativa física e documentada.
2. **Descartar sujeitos com base em performance futura** — nesta etapa só critérios de qualidade objetivos (tempo, buracos, contagem).
3. **Usar índice da linha como tempo** — sempre a coluna time real.
4. **Assumir 60 Hz sem medir** — sempre estime fs dos dados; aqui confirmou 60 Hz.
5. **Ignorar sujeito com gravação curta** — sujeito 26 (~22 s) gera menos janelas; anotar, não excluir automaticamente.
6. **Confundir alinhamento com normalização** — alinhamento é temporal; normalização é escala (Etapa 6).
7. **Plotar só um sujeito** — padrões variam; inspecione pelo menos 3.
8. **Corrigir dados olhando o grupo de teste** — quando o split existir (Etapa 5), decisões de limpeza devem usar só o grupo de desenvolvimento.

---


---


# Etapa 5 — Separação externa 70/30 por sujeito

## 1. O conceito, de forma simples

Até aqui você carregou e inspecionou **todos** os sujeitos juntos. Agora separa **quem serve para desenvolver o modelo** de **quem serve para avaliação final**.

A regra é uma só: o split é por **sujeito**, nunca por linha ou janela.

- **70% → desenvolvimento** — normalização, LOSO, escolha de hiperparâmetros e treino interno
- **30% → teste final** — intocado até a Etapa 15

Cada sujeito inteiro vai para um único grupo. Nenhum subject_id aparece nos dois.

---

## 2. Por que isso importa no seu problema

Sinais IMU de um mesmo indivíduo compartilham postura, ritmo, telefone e contexto. Se janelas do sujeito 02 entram no treino e no teste, o modelo **já viu padrões daquele corpo** — a métrica fica otimista e não mede generalização.

| Split errado | Consequência |
|---|---|
| Por janela/amostra | Vazamento entre treino e teste |
| Normalizar com todos os sujeitos | Estatísticas do teste influenciam o treino |
| Ajustar hiperparâmetros olhando o teste | Overfitting ao conjunto de avaliação |
| Não salvar a lista de sujeitos | Impossível reproduzir o experimento |

O grupo de **teste final** definido agora permanece **congelado** até a Etapa 15.

---

---

---

## 5. Resultado na sua base real

| Grupo | Sujeitos | Fração |
|---|---|---|
| **Desenvolvimento** | 02, 03, 08, 09, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 27, 28, 29 | 19/27 = **70,4%** |
| **Teste final** | 01, 04, 06, 10, 11, 23, 24, 26 | 8/27 = **29,6%** |

Semente: 42 (mesma da Etapa 1).

Arquivos salvos:
- outputs/pipeline_dl/splits/external_split_70_30.json
- outputs/pipeline_dl/splits/dev_subject_ids.txt
- outputs/pipeline_dl/splits/test_subject_ids.txt

---

## 6. Principais erros a evitar

1. **Split por janela** — janelas do mesmo sujeito nos dois grupos invalidam a avaliação.
2. **Recalcular o split a cada execução sem salvar** — use os arquivos em splits/.
3. **Olhar métricas do teste antes da Etapa 15** — qualquer ajuste motivado pelo teste é vazamento.
4. **Normalizar agora usando todos os sujeitos** — normalização vem na Etapa 6, **só com dados de treino de cada fold**.
5. **Criar janelas antes do split** — grupos de sujeitos devem estar definidos primeiro (feito aqui).
6. **Mudar a semente depois** — altera quem vai para teste; documente e mantenha fixa.
7. **Excluir sujeito 26 por ser mais curto** — decisões de exclusão devem usar critérios objetivos e, depois do split, **apenas no grupo dev**.

---


---


# Etapa 6 — Normalização sem vazamento

## 1. O conceito, de forma simples

Redes neurais aprendem melhor quando as entradas estão em escalas parecidas. **Normalizar** transforma cada canal (acc_x, acc_y, …) para algo próximo de média 0 e desvio 1.

A regra crítica: o escalador aprende parâmetros (média e desvio) **somente com dados de treino do fold**. Validação e teste recebem a **mesma transformação**, sem participar do ajuste do escalador.

---

## 2. Por que isso importa no seu problema

### Por que normalizar com todos os sujeitos é erro

Se você calcula média/desvio usando **todos** os 27 sujeitos (incluindo teste final):

- estatísticas do grupo de **teste** influenciam o treino;
- métricas de validação ficam **otimistas**;
- o modelo não simula uso real (novo sujeito nunca visto).

### No seu pipeline

| Momento | Quem entra no ajuste do escalador |
|---|---|
| Cada fold LOSO (Etapas 6–13) | 18 sujeitos dev de treino (ex.: todos exceto 02) |
| Validação do fold | 1 sujeito dev (ex.: 02) — só transform() |
| Teste final (Etapa 15) | 19 sujeitos dev — **nunca** os 8 de teste no fit |

### Normalizar o alvo vicon?

| Opção | Vantagem | Desvantagem |
|---|---|---|
| **Sim** (recomendado para treino) | Loss numérica mais estável | Métricas precisam voltar para cm |
| **Não** | MAE/RMSE direto em cm | Escala do alvo pode dominar a loss |

Se normalizar o vicon, use reversão da normalização do alvo antes de reportar MAE clínico em centímetros.

---

---

---

## 5. Resultado no seu fold de exemplo

Fold: **LOSO com sujeito 02 fora** (18 sujeitos dev no treino)

**Treino após normalizar** (deve ficar ~0 e ~1):

| canal | média | desvio |
|---|---|---|
| acc_x … gyro_z | ≈ 0,0000 | ≈ 1,0000 |

**Validação (02)** com parâmetros do treino (média pode ≠ 0):

| canal | média | desvio |
|---|---|---|
| acc_x | -0,53 | 0,75 |
| acc_z | 1,59 | 0,57 |
| … | … | … |

Isso indica que o sujeito 02 tem perfil de sinal diferente da média do treino — comportamento normal em LOSO.

**Inversão do alvo** (3 pontos ilustrativos):
- escalado: [-0.899, -0.901, -0.903]
- em cm: [0.000, -0.004, -0.013]

Arquivos salvos:
- outputs/pipeline_dl/scalers/loso_val_02_scaler.joblib
- outputs/pipeline_dl/scalers/loso_val_02_meta.json

---

## 6. Principais erros a evitar

1. **ajuste do escalador com todos os sujeitos** — incluindo validação ou teste final.
2. **Recalcular scaler por sujeito** — deve ser por **fold de treino**, não por indivíduo.
3. **Normalizar time** — tempo não é feature; não entra no escalador.
4. **Esquecer inverse_transform** — se vicon foi escalado, MAE em cm exige reversão.
5. **Usar estatísticas do teste na Etapa 15** — scaler final: fit só nos 19 dev.
6. **Misturar normalização e janelamento** — ordem correta: normalizar séries contínuas, **depois** criar janelas (Etapa 7).
7. **Salvar um único scaler global** — em LOSO, cada fold tem seu próprio pacote de escaladores.

---


---


# Etapa 7 — Criação de janelas temporais

## 1. O conceito, de forma simples

Até aqui cada sujeito é uma **série contínua** com milhares de linhas. Redes temporais (CNN 1D, TCN) aprendem a partir de **pedaços finitos** dessa série — as **janelas**.

Exemplo com seus parâmetros:

| Parâmetro | Valor | Significado |
|---|---|---|
| Frequência | 60 Hz | 60 amostras por segundo |
| Janela | 2 s | 120 amostras por janela |
| Stride | 1 s | avança 60 amostras a cada nova janela |
| Entrada X | 6 × 120 | 6 canais IMU × 120 instantes |
| Alvo y | 1 número | amplitude = max(vicon) − min(vicon) na janela |

**Stride** controla a sobreposição: stride de 1 s com janela de 2 s → **50% de sobreposição** (cada janela compartilha metade das amostras com a seguinte).

---

## 2. Por que isso importa no seu problema

O modelo não vê a gravação inteira de uma vez — vê janelas de 2 s e precisa inferir quanto o Vicon se moveu **naquele intervalo** a partir do padrão IMU.

| Decisão | Impacto |
|---|---|
| Tamanho da janela | Muito curta → pouco contexto; muito longa → movimentos distintos misturados |
| Stride | Stride grande → menos janelas; stride pequeno → mais janelas, mais correlação entre elas |
| Amplitude vs curva | Amplitude é mais simples; curva exige prever 120 valores por janela |
| subject_id por janela | Permite split/LOSO sem misturar sujeitos |

**Ordem correta no pipeline:**
1. Split por sujeito (Etapa 5)
2. Normalizar treino do fold + transformar val (Etapa 6)
3. **Criar janelas** separadamente em treino e val (esta etapa)

---

---

---

## 5. Resultado no fold de exemplo (LOSO, val = sujeito 02)

| Conjunto | Shape X | Shape y | Janelas |
|---|---|---|---|
| Treino (18 sujeitos dev) | (573, 6, 120) | (573,) | ~31–32 por sujeito |
| Validação (sujeito 02) | (32, 6, 120) | (32,) | 32 |

Parâmetros confirmados:
- Janela: **2 s = 120 amostras**
- Stride: **1 s = 60 amostras**
- Sobreposição: **50%**
- Modo: **amplitude**

Exemplo de alvos (vicon escalado, sujeito 02):

| Janela | t₀ | y (amplitude escalada) |
|---|---|---|
| 0 | 0,000 s | 2,2509 |
| 1 | 1,000 s | 2,2509 |
| 2 | 2,000 s | 2,2566 |

---

## 6. Principais erros a evitar

1. **Janelar antes do split** — define grupos de sujeitos primeiro (Etapa 5).
2. **Misturar janelas de treino e val de sujeitos diferentes sem controle** — mantenha lotes separados por fold.
3. **Perder subject_id** — sem ele, shuffle no carregador de batches pode causar vazamento indireto.
4. **Layout errado para PyTorch** — use (n_janelas, canais, tempo), não (n_janelas, tempo, canais).
5. **Confundir stride com tamanho da janela** — stride = passo entre janelas; window = tamanho de cada uma.
6. **Começar com modo curve** — amplitude escalar é o ponto de partida mais sólido.
7. **Janelar sujeito com gravação curta sem verificar** — se n_amostras < 120, o sujeito gera 0 janelas (sujeito 26 ainda tem 1325 amostras → OK).
8. **Calcular amplitude em cm e treinar em escala normalizada misturadas** — mantenha coerência entre y e o pacote de escaladores do fold.

---


---


# Etapa 8 — Dataset e carregador de batches no PyTorch

## 1. O conceito, de forma simples

Na Etapa 7 você produziu arrays NumPy com janelas. Agora eles entram no ecossistema PyTorch:

- **Dataset** — define como acessar **uma janela** por vez (acesso a cada amostra)
- **carregador de batches** — agrupa janelas em **mini-batches**, opcionalmente embaralha, e entrega tensores prontos para o modelo

Fluxo de um batch de treino:

---

## 2. Por que isso importa no seu problema

| Componente | Função no pipeline |
|---|---|
| IMUWindowDataset | Padroniza o que o modelo recebe por amostra |
| collate_fn | Empilha janelas num batch com shape correto |
| embaralhamento ativado no treino (treino) | Mistura janelas **dentro** do conjunto de treino |
| sem embaralhamento na validação (val/test) | Ordem fixa e reprodutível |
| subject_id no batch | Rastreabilidade por janela (métricas por sujeito) |

O ponto crítico: **shuffle mistura janelas, não sujeitos entre conjuntos**. Treino tem sujeitos {03, 08, …}; validação só {02}. Isso foi garantido nas Etapas 5–7 e verificado aqui.

---

---

---

## 5. Resultado no fold de exemplo (LOSO, val = 02)

| | Treino | Validação |
|---|---|---|
| Amostras (janelas) | 573 | 32 |
| Batches (batch=32) | 18 | 1 |
| Sujeitos | 18 dev | só 02 |
| shuffle | True | False |

**Mini-batch de treino:**

**Mini-batch de validação:**

---

## 6. Principais erros a evitar

1. **Um único carregador de batches para treino + val** — sempre separe por conjunto.
2. **embaralhamento ativado no treino na validação** — métricas ficam não reprodutíveis.
3. **Shape (batch, tempo, canais)** — Conv1d exige (batch, canais, tempo).
4. **Esquecer float32** — use float32 para x e y (GPU e memória).
5. **Descartar subject_id** — necessário para métricas por sujeito e auditoria de vazamento.
6. **Batch size maior que o conjunto de val** — com 32 janelas de val e batch=32, há 1 batch; OK aqui, mas com menos janelas ajuste o batch size.
7. **Misturar sujeitos de teste final no treino** — teste final só entra na Etapa 15, com loader próprio e sem embaralhamento na validação.
8. **Confundir shuffle com split** — shuffle não substitui split por sujeito; só reordena janelas já separadas.

---


---


# Etapa 9 — Modelo inicial: CNN 1D

## 1. O conceito, de forma simples

Uma **CNN 1D** desliza filtros ao longo do **tempo** (não da imagem) para detectar padrões locais no sinal IMU — picos de aceleração, oscilações, transients — e resume a janela inteira num **único número**: a amplitude predita do Vicon.

---

## 2. Por que isso importa no seu problema

Integrar aceleração analiticamente falha por drift e orientação. A CNN aprende uma **relação empírica** entre o padrão temporal do IMU e a amplitude medida pelo Vicon.

Por que começar com CNN 1D:

| Vantagem | No contexto IMU → Vicon |
|---|---|
| Padrões locais | Impulsos e ciclos de movimento ficam visíveis para filtros temporais |
| Multicanal | Conv1d mistura acc e gyro no mesmo filtro |
| Eficiência | Mesmos pesos ao longo da janela — menos parâmetros que LSTM grande |
| Baseline sólida | Se TCN não superar CNN 1D, a complexidade extra pode não valer |

---

---

---

## 5. Resultado do forward pass (modelo não treinado)

Com mini-batch real do fold LOSO (val = sujeito 02):

Isso é esperado: a Etapa 9 só valida a **arquitetura e o fluxo de tensores**. O treino vem na Etapa 12.

---

## 6. Principais erros a evitar

1. **Entrada (batch, tempo, canais)** — Conv1d exige (batch, canais, tempo).
2. **Esquecer .squeeze(-1)** — saída (batch, 1) quebra a loss escalar.
3. **Kernel maior que a janela após pooling** — após 3 poolings, tempo = 15; kernels 3–7 ainda funcionam.
4. **Treinar agora sem loop completo** — loss, optimizer e early stopping são Etapa 11–12.
5. **Comparar predição escalada com MAE em cm** — use reversão da normalização do alvo se o vicon foi normalizado.
6. **Pular CNN 1D e ir direto para TCN** — CNN 1D é a baseline de referência (Etapa 10).
7. **BatchNorm com batch=1 na inferência** — use modo de avaliação do modelo; no treino com batches pequenos, considere ajustar batch size.

---


---


# Etapa 10 — Modelo principal: TCN

## 1. O conceito, de forma simples

A **TCN (Temporal Convolutional Network)** usa convoluções **dilatadas** empilhadas para enxergar **mais contexto temporal** sem reduzir a resolução com pooling.

- **Dilatação 1** — filtro olha vizinhos imediatos  
- **Dilatação 2** — salta 1 amostra entre pesos  
- **Dilatação 4, 8, 16…** — alcance cresce rapidamente  

Cada **bloco temporal** tem duas convoluções dilatadas + **conexão residual** (entrada + saída), o que facilita treinar pilhas mais profundas.

A resolução temporal permanece **120 amostras** em toda a TCN — diferente da CNN 1D, que comprime com MaxPool.

---

## 2. Por que isso importa no seu problema

Movimentos corporais em 2 s podem ter **fases distantes** relacionadas (aceleração inicial ↔ pico de deslocamento no Vicon). A TCN foi pensada para isso:

| | **CNN 1D (Etapa 9)** | **TCN (Etapa 10)** |
|---|---|---|
| Contexto | Local; pooling reduz tempo | Longo; dilatação expande alcance |
| Resolução temporal | 120 → 60 → 30 → 15 | 120 em todos os blocos |
| Campo receptivo | ~15–30 amostras finais | **125 amostras** (~2,08 s) |
| Parâmetros | ~45 k | ~85 k |
| Risco | Underfitting de contexto | Overfitting se poucos dados |

**Quando a TCN tende a ser melhor:** padrões que dependem de 1–2 s de contexto, relação entre fases distantes do movimento.

**Quando a CNN 1D basta:** padrões muito locais, poucos dados, baseline mais rápida.

---

---

---

## 5. Resultado do forward pass (modelo não treinado)

| Métrica | Valor |
|---|---|
| Entrada | (32, 6, 120) |
| Saída | (32,) |
| Parâmetros | 84 961 |
| Campo receptivo | **125 amostras (2,08 s)** |
| Cobre janela inteira | **Sim** |

**Evolução das shapes:**

Predições ainda aleatórias — treino vem na Etapa 12.

---

## 6. Principais erros a evitar

1. **Dilatações curtas demais** — com [1,2,4,8] o RF = 61 amostras (< 120); não cobre a janela inteira.
2. **Confundir causal com offline** — aqui usamos TCN causal; para janelas completas offline, bidirecional ou não-causal também é opção (Etapa 14).
3. **Assumir TCN > CNN 1D sempre** — compare no LOSO (Etapa 13); mais parâmetros podem overfitar com ~573 janelas de treino.
4. **Esquecer Chomp1d** — sem ele, shapes temporais ficam desalinhadas.
5. **Não parear dilatações com num_channels** — listas devem ter o mesmo comprimento.
6. **Treinar agora sem loss/métricas** — Etapa 11 define MSE, Adam e MAE/RMSE/R².

---


---


# Etapa 11 — Função de perda, otimizador e métricas

## 1. O conceito, de forma simples

Treinar uma rede envolve duas coisas distintas:

1. **Loss (perda)** — o que o modelo **otimiza** via gradiente (retropropagação)
2. **Métricas** — o que você **interpreta e reporta** (MAE em cm, R², bias clínico)

A loss guia o aprendizado; as métricas respondem: *“o erro é aceitável?”*

---

## 2. Por que isso importa no seu problema

Regressão de amplitude Vicon a partir de IMU costuma ter:

- erros ocasionais grandes (janelas atípicas)
- alvo possivelmente **normalizado** (Etapa 6)
- necessidade de reportar erro em **unidade clínica** (cm)

| Loss | Comportamento | Quando usar |
|---|---|---|
| **MSE** | Penaliza erros grandes ao quadrado | Baseline; sensível a outliers |
| **MAE (L1)** | Robusta, gradiente instável em 0 | Poucos outliers extremos |
| **Huber** | Quadrática perto de 0, linear longe | **Boa candidata padrão** — robusta e suave |

**Adam** + **weight decay** equilibram convergência e regularização — útil porque a TCN tem ~85 k parâmetros vs ~45 k da CNN 1D.

---

---

---

## 5. Resultado da demonstração

**Exemplo ilustrativo (5 pontos fictícios):**

| Métrica | Valor |
|---|---|
| MAE | 0,32 |
| RMSE | 0,34 |
| R² | 0,94 |
| Bias | +0,12 |
| MAPE | 12,4% |

**TCN não treinada — fold LOSO (vicon escalado):**

| | Treino | Validação |
|---|---|---|
| Loss (Huber) | 2,04 | 1,85 |
| MAE | 2,54 | 2,35 |
| RMSE | — | 2,36 |
| R² | — | −314 (esperado: modelo aleatório) |
| Bias | — | −2,35 |

O R² muito negativo **antes do treino** é normal — confirma que as métricas funcionam e que o modelo ainda não aprendeu.

---

## 6. Principais erros a evitar

1. **Reportar loss como MAE clínico** — são grandezas diferentes.
2. **MAE em escala normalizada sem reverter** — use reversão da normalização do alvo para cm.
3. **MAPE com amplitudes ~0** — distorce percentual; interprete com cuidado.
4. **Só olhar R²** — complemente com MAE, bias e erro por sujeito.
5. **MSE pura com outliers** — considere Huber (padrão do pipeline).
6. **Learning rate alto demais** — comece com 1e-3; ajuste no LOSO (Etapa 14).
7. **Esquecer modo de avaliação do modelo na validação** — BatchNorm se comporta diferente em treino vs eval.
8. **Confundir métricas de treino com generalização** — val LOSO é a referência interna.

---


---


# Etapa 12 — Treinamento de um único fold

## 1. O conceito, de forma simples

Até aqui você montou dados, modelo, loss e métricas. Agora roda o **loop de aprendizado** para **um fold LOSO** dentro dos 70% de desenvolvimento:

Se a validação não melhorar por patience épocas, o treino para e restaura os **melhores pesos**.

---

## 2. Por que isso importa no seu problema

Treinar **um fold por vez** antes do LOSO completo permite:

| Objetivo | Benefício |
|---|---|
| Verificar se loss desce | Pipeline inteiro funciona |
| Inspecionar curvas treino/val | Detectar overfitting cedo |
| Early stopping | Evita memorizar sujeitos de treino |
| Salvar checkpoint | Reprodutibilidade e Etapa 13 reutiliza o padrão |

Nesta etapa usa-se **apenas sujeitos dev** (fold LOSO com val = 02). O **teste final (30%) continua intocado**.

---

---

---

## 5. Resultado do treino real (fold val = sujeito 02)

| Métrica | Antes (Etapa 11) | Depois (Época 14) |
|---|---|---|
| Val loss (Huber) | 1,85 | **0,017** |
| Val MAE | 2,35 | **0,144** (vicon escalado) |
| Val RMSE | 2,36 | **0,182** |
| Épocas | — | 26 (early stop na 26, melhor na **14**) |

Treino: **573 janelas** (18 sujeitos dev) | Val: **32 janelas** (sujeito 02)

Arquivos gerados:
- outputs/pipeline_dl/checkpoints/etapa12_loso_val_02_best.pt
- pasta de gráficos do experimentoetapa12_loss_loso_val_02.png
- outputs/pipeline_dl/metrics/etapa12_loso_val_02_metrics.json

**Sobre R² negativo na validação:** com apenas 32 janelas de um único sujeito e alta variância local, R² pode ficar instável. MAE e RMSE são mais confiáveis neste fold pequeno. No LOSO completo (Etapa 13), agregando todos os sujeitos, R² ganha interpretação.

---

## 6. Principais erros a evitar

1. **Treinar sem modo de avaliação do modelo na validação** — BatchNorm distorce métricas.
2. **Escolher melhor modelo pelo train_loss** — use **val_loss** (early stopping).
3. **Não salvar checkpoint** — perde melhor época após early stopping.
4. **Incluir sujeito de teste final** — só grupo dev nesta etapa.
5. **Confundir épocas executadas com melhor época** — aqui: 26 executadas, melhor na 14.
6. **Reportar MAE escalado como cm** — aplique reversão da normalização do alvo do pacote de escaladores.
7. **Rodar LOSO completo agora** — isso é Etapa 13.
8. **Ignorar curvas** — val_loss subindo enquanto train desce → overfitting.

---


---


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


---


# Etapa 14 — Escolha de hiperparâmetros (pronta para rodar depois)

---

## 1. O conceito, de forma simples

Você já tem LOSO (Etapa 13) para medir generalização **dentro dos 70% dev**. A Etapa 14 **compara combinações de hiperparâmetros** com o mesmo protocolo e escolhe a que tem **menor MAE média** entre os folds.

O **teste final (30%) continua intocado**.

---

## 2. O que entra na busca

| Hiperparâmetro | O que testa |
|---|---|
| window_seconds | Quanto contexto temporal (1s, 2s, 3s) |
| stride_seconds | Sobreposição entre janelas |
| model_type | tcn vs cnn1d |
| loss | Huber vs MSE |
| learning_rate | Velocidade de aprendizado |
| batch_size | Estabilidade do gradiente |

**Critério de seleção:** menor **MAE média** no LOSO dev (não loss de treino).

Grid padrão com **8 trials** (tcn_baseline, cnn1d, win_1s, win_3s, loss_mse, lr_5e4, batch_16, stride_05).

---

---

---

## 5. Fluxo recomendado

1. **Etapa 13** (opcional) — LOSO baseline com config padrão  
2. **Etapa 14 LOSO rápido com 3 sujeitos** — triagem rápida  
3. **Revalidar o vencedor** — triagem rápida com poucos folds; depois LOSO completo no trial escolhido  
4. **Etapa 15** — treino final nos 70% + teste nos 30% com best_hyperparameters.json

---

## 6. Erros a evitar

1. **Usar teste final para escolher hiperparâmetros** — proibido até Etapa 15  
2. **Confiar só na triagem rápida** — revalide o vencedor com LOSO completo  
3. **Comparar trials com números de folds diferentes** — compare MAE só com mesmo N de folds  
4. **Grid enorme sem triagem** — comece com LOSO rápido com 3 sujeitos  
5. **Ignorar best_hyperparameters.json** — Etapa 15 deve carregar esse arquivo

---


---


# Etapa 15 — Treino final + teste intocado (pronta para rodar depois)

---

## O que faz

1. **Scaler** ajustado só nos **19 sujeitos dev** (70%)
2. **Teste (8 sujeitos, 30%)** recebe apenas transform() — uma única vez
3. **Treino final** em **605 janelas** dev (100%, sem validação interna)
4. **Avaliação única** em **243 janelas** de teste
5. Métricas em escala normalizada **e em cm** (via inverse_transform)
6. Salva modelo, scaler, predições e métricas

---

## Prévia do split (modo prévia)

| Grupo | Sujeitos | Janelas |
|---|---|---|
| Dev (treino) | 19 | 605 |
| Teste (intocado) | 01, 04, 06, 10, 11, 23, 24, 26 | 243 |

Épocas número automático de épocas: mediana do LOSO Etapa 13 = **6** (pode sobrescrever com 14 épocas manualmente se preferir).

---

---

---

## Regras importantes

- **Não reajuste** hiperparâmetros depois de olhar o teste
- Rode a **Etapa 14** antes se quiser comparar configs (usar melhores hiperparâmetros salvos)
- O LOSO da Etapa 13 já está pronto como baseline dev (**MAE ≈ 0,21** normalizado)

---


---


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


---


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

Estas etapas foram **planejadas** como extensão natural do pipeline, mas **ainda não existem** como módulos implementados. O pipeline oficial termina na Etapa 16.

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
