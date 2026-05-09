# Roteiro de Apresentação Final
## NYC Rideshare Fare Prediction — Big Data

> Este documento é o roteiro completo para montar os slides e conduzir a apresentação.
> Cada seção indica o que mostrar, o que falar e quais dados/prints usar.

---

## Estrutura geral sugerida (15–20 min)

| # | Slide | Tempo |
|---|-------|-------|
| 1 | Capa | — |
| 2 | Problema e motivação | 1 min |
| 3 | Dataset | 2 min |
| 4 | Arquitetura do ambiente | 1 min |
| 5 | Pipeline de dados — limpeza e features | 2 min |
| 6 | EDA — principais achados | 2 min |
| 7 | Features engineered | 1 min |
| 8 | Modelo 1: Linear Regression | 1.5 min |
| 9 | Modelo 2: Decision Tree | 1.5 min |
| 10 | Modelo 3: Neural Network | 2 min |
| 11 | Comparativo final dos modelos | 2 min |
| 12 | Desafios técnicos enfrentados | 2 min |
| 13 | Melhorias futuras | 1 min |
| 14 | Conclusão | 1 min |

---

## Slide 1 — Capa

**Título:** Previsão de Tarifa de Rideshare em Nova York  
**Subtítulo:** Pipeline Big Data com PySpark, MLlib e PyTorch  
**Integrantes**, disciplina, data

---

## Slide 2 — Problema e motivação

**O que mostrar:**
- "Quanto vai custar minha corrida de Uber ou Lyft em NYC?"
- Tarifa não é determinística: varia por distância, horário, zona, operadora, demanda
- Valor de negócio: transparência para o passageiro, planejamento para motoristas, modelagem de surge pricing

**O que falar:**
- O target é `passenger_fare` — a tarifa base antes de gorjeta e taxas
- Por que não usar `driver_pay`? É regulatório (TLC Local Law 150/2018): calculado por fórmula, R² ~0.97 trivialmente. Não há aprendizado real.
- Escolhemos `passenger_fare` porque carrega sinal de mercado real (surge, zona, demanda)

---

## Slide 3 — Dataset

**O que mostrar:**

```
Fonte: Kaggle "NYC Rideshare Raw Data" (aaronweymouth)
Origem primária: NYC Taxi & Limousine Commission (HVFHV)
Tamanho: ~365M linhas, ~12 GB (Dez/2021–Ago/2023)
Recorte usado: Março–Agosto 2023 (6 meses)
Operadoras: Uber e Lyft
```

Tabela das colunas principais (5–6 colunas, não todas):
| Coluna | Descrição |
|--------|-----------|
| `passenger_fare` | **TARGET** — tarifa base (USD) |
| `trip_length` | Distância em milhas |
| `total_ride_time` | Duração em segundos |
| `hour_of_day` | Hora do embarque |
| `business` | Uber ou Lyft |
| `date` | Data do embarque |

**Highlight:** colunas de leakage excluídas — `hourly_rate` e `dollars_per_mile` são derivadas do target; `driver_total_pay` é regulatório.

**Split temporal:**
```
Treino: Março–Maio 2023  →  56.3M linhas
Teste:  Junho–Agosto 2023 →  53.8M linhas
SPLIT_DATE = "2023-06-01"
```
Por que temporal e não aleatório? Evita data leakage — o modelo nunca vê o futuro durante o treino.

---

## Slide 4 — Arquitetura do ambiente

**Diagrama:**
```
┌─────────────────────────────────────────────────────┐
│                    Docker Compose                   │
│                                                     │
│  spark-master  ←→  spark-worker-1  (4GB / 2 cores) │
│                ←→  spark-worker-2  (4GB / 2 cores) │
│                                                     │
│  jupyter (driver)  ←  Volume: ./data:/data          │
│  localhost:8888        ./models:/models             │
│                        ./results:/results           │
└─────────────────────────────────────────────────────┘
```

**O que falar:**
- Cluster Spark standalone em Docker — reproduzível em qualquer máquina com Docker
- Sem HDFS, sem MinIO: simplicidade de setup para ambiente acadêmico
- Volume local compartilhado entre todos os containers (driver + workers enxergam o mesmo `/data`)
- PySpark 3.5.8 alinhado entre driver e workers — versão incompatível causa falha de serialização (foi um dos bugs enfrentados)

**Print sugerido:** Spark UI em `localhost:8080` mostrando 2 workers ALIVE

---

## Slide 5 — Pipeline de dados: limpeza e features

**Diagrama de fluxo:**
```
rideshare_data.parquet (365M linhas)
        │
        ▼ filtro temporal (Mar–Ago 2023)
   trips_raw (116.2M linhas)
        │
        ▼ limpeza por percentil + feature engineering
   trips_clean (110.1M linhas)  ← 5.26% drop (DQ gate)
        │
        ▼ particionado por pickup_year_month
   /data/silver/trips_silver/   (parquet em disco)
```

**Limpeza (Spark SQL puro):**
- Filtros por percentis 0.1%–99.9% em `passenger_fare`, `trip_length`, `speed_mph` — distribuições heavy-tailed, cutoffs fixos seriam arbitrários
- Remoção de zonas inválidas 264 e 265 ("Unknown", "Outside NYC")
- `request_to_pickup >= 0` (espera negativa = erro de timestamp)
- `speed_mph > 0.5` (GPS travado ou corrida cancelada mascarada)
- DQ gate: `assert pct_removido < 20%` — aborta se limpeza remover dados demais

**O que falar sobre a escolha de percentis:**
> "Velocidade média em Manhattan no horário de pico é ~7 mph. Um cutoff fixo de '1–80 mph' eliminaria dados válidos. Percentis se calibram automaticamente para a distribuição real de NYC."

**Print sugerido:** output da célula de audit do notebook 02 mostrando linhas por mês e % removido

---

## Slide 6 — EDA: principais achados

Escolher 3–4 dos achados mais interessantes do notebook 01. Plots disponíveis (ver `01_eda.ipynb`):
- Histograma da tarifa em escala linear + log
- Tarifa média e mediana por hora do dia (com bandas de rush/madrugada)
- Heatmap de tarifa por dia da semana × hora
- Boxplot Uber vs Lyft (tarifa por bucket de distância)
- Tarifa média e volume por borough
- Hexbin `trip_length × passenger_fare`
- Volume de corridas por hora e top 10 zonas de embarque
- Matriz OD por borough

**Achado 1 — Distribuição da tarifa**
- Mediana ~$17–22, cauda longa até >$100 (corridas para aeroporto)
- Justifica log-transform na NN e filtros de percentil

**Achado 2 — Variação por hora do dia**
- Tarifa média mais alta: madrugada (0h–4h) e horário de pico (7h–9h, 16h–19h)
- Justifica as features `is_rush_hour` e `is_late_night`

**Achado 3 — Variação por borough**
- Manhattan: corridas mais longas e caras
- Bronx: corridas mais curtas e baratas
- Justifica `pu_borough` e `do_borough` como features categóricas

**Achado 4 — Correlação distância × tarifa**
- Correlação mais forte entre `trip_length` e `passenger_fare` (~0.88)
- Mas não linear: surge pricing, rotas de aeroporto, etc.

---

## Slide 7 — Feature engineering

**Tabela compacta (não listar todas, destacar as mais importantes):**

| Feature | Como foi gerada | Por que importa |
|---------|----------------|-----------------|
| `speed_mph` | `trip_length / (total_ride_time / 3600)` | Proxy de congestionamento |
| `is_rush_hour` | hora 7–9 ou 16–19 | Surge pricing em pico |
| `is_late_night` | hora >= 22 ou <= 4 | Tarifa noturna |
| `pickup_airport` | zona IN (1, 132, 138) | Corridas EWR/JFK/LGA têm tarifa fixa/maior |
| `same_borough` | pu_borough = do_borough | Corridas curtas intra-bairro |
| `pu_borough` / `do_borough` | join com taxi_zone_lookup.csv | Variação geográfica de preço |

**Total: 14 numéricas + 4 categóricas = 18 features**

---

## Slide 8 — Modelo 1: Linear Regression (MLlib)

**O que mostrar:**

```
Algoritmo:  LinearRegression (Spark MLlib)
Pipeline:   StringIndexer → VectorAssembler → StandardScaler → LR
Treino:     56.3M linhas (conjunto completo)
```

**Métricas:**
| Métrica | Valor |
|---------|-------|
| R² | 0.806 |
| RMSE | $9.07 |
| MAE | $5.76 |
| Tempo treino | 229s |

**Plots disponíveis (`03_linear_regression.ipynb`):**
- Coeficientes (interpretáveis em USD por desvio padrão)
- Predito vs Real em hexbin (densidade próxima da diagonal)
- RMSE/MAE por borough
- RMSE/MAE por bucket de distância
- Resíduo médio por hora do dia + MAE
- Histograma e scatter de resíduos

**Análise estratificada:**
- Erro maior em Manhattan (corridas mais variáveis) e viagens 10+ mi
- LR captura tendência linear mas não surge pricing não-linear

**O que falar:**
- Baseline interpretável: coeficientes mostram que distância e aeroporto são os preditores mais fortes
- StandardScaler essencial — sem escala as features de segundos (30.000) dominam sobre booleanos (0/1)

---

## Slide 9 — Modelo 2: Decision Tree (MLlib)

**O que mostrar:**

```
Algoritmo:  DecisionTreeRegressor (Spark MLlib)
Parâmetros: maxDepth=8, maxBins=256
Pipeline:   StringIndexer → VectorAssembler → VectorIndexer → DT
Treino:     56.3M linhas (conjunto completo)
```

**Métricas:**
| Métrica | Valor |
|---------|-------|
| R² | 0.813 |
| RMSE | $8.91 |
| MAE | $5.13 |
| Tempo treino | 436s |

**Plots disponíveis (`04_decision_tree.ipynb`):**
- Importância das features (top 15)
- Predito vs Real em hexbin
- RMSE/MAE por borough
- RMSE/MAE por bucket de distância
- Resíduo médio por hora × operadora (Uber vs Lyft)
- Histograma e scatter de resíduos

Top features esperadas: `trip_length`, `speed_mph`, `total_ride_time`, `pickup_airport`

**O que falar:**
- Melhor R² e RMSE entre os 3 modelos
- Captura não-linearidades (surge pricing, tarifas fixas de aeroporto) que o LR não consegue
- `VectorIndexer(maxCategories=8)`: sem isso, o DT trata índices de borough como ordenados (borough_idx=0 < 1 < 2), fazendo splits ordinais incorretos. Com VectorIndexer, ele sabe que são categorias e testa todos os subconjuntos.
- Restrição do enunciado respeitada: Decision Tree, não GBT nem Random Forest

---

## Slide 10 — Modelo 3: Neural Network (PyTorch)

**O que mostrar:**

```
Framework:    PyTorch (não distribuído)
Arquitetura:  256 → BN → ReLU → Dropout(0.2)
              128 → BN → ReLU → Dropout(0.2)
              64  → ReLU → 1
Loss:         MSELoss em log(passenger_fare)
Optimizer:    Adam (lr=1e-3) + ReduceLROnPlateau
Dados:        1.28M linhas estratificadas Mar+Abr+Mai (~427k/mês)
```

**Métricas:**
| Métrica | Valor |
|---------|-------|
| R² | **0.815** |
| RMSE | **$8.91** |
| MAE | **$4.94** |
| Tempo treino | 555s |

**Plots disponíveis (`05_neural_network.ipynb`):**
- Curva de loss train/val + LR schedule (ReduceLROnPlateau)
- Predito vs Real em hexbin
- RMSE/MAE por borough
- RMSE/MAE por bucket de distância
- Boxplot de resíduo por bucket de distância × operadora
- Histograma e scatter de resíduos

**O que falar:**
- NN tem o **menor MAE** ($4.94) e o **maior R²** ($0.815) — empata com DT em RMSE e supera em todos os outros critérios
- Log-transform do target foi decisivo: MAE caiu de $6.89 → $4.94
- Amostragem estratificada (Mar+Abr+Mai com peso igual) supera amostragem aleatória, que enviesa para os primeiros meses
- Limitação de hardware: PyTorch não é distribuído. `toPandas()` de 5M linhas causou OOM. Viável na máquina de 8GB: 1.28M linhas (~2% do treino total)
- Justificativa metodológica: o objetivo é demonstrar capacidade do modelo, não throughput

---

## Slide 11 — Comparativo final

**Tabela principal:**

| Modelo | R² | RMSE | MAE | Treino | Dados |
|--------|----|------|-----|--------|-------|
| Linear Regression | 0.806 | $9.07 | $5.76 | 229s | 56.3M |
| Decision Tree | 0.813 | $8.91 | $5.13 | 436s | 56.3M |
| **Neural Network** | **0.815** | **$8.91** | **$4.94** | 555s | 1.28M |

**Análise estratificada por borough (Decision Tree):**
| Borough | RMSE | MAE |
|---------|------|-----|
| EWR | $80.47 | $65.55 |
| Manhattan | $10.54 | $6.58 |
| Queens | $9.16 | $4.98 |
| Brooklyn | $7.20 | $4.03 |
| Staten Island | $6.74 | $3.49 |
| Bronx | $5.70 | $3.29 |

**Análise por distância (Decision Tree):**
| Bucket | RMSE | MAE |
|--------|------|-----|
| 0–2 mi | $4.77 | $2.81 |
| 2–5 mi | $6.79 | $4.31 |
| 5–10 mi | $10.78 | $7.08 |
| 10+ mi | $17.17 | $11.56 |

**O que falar:**
- EWR tem RMSE $80: apenas 3 corridas no teste, estatisticamente irrelevante
- Corridas 10+ mi têm RMSE 3.6× maior que 0–2 mi: surge pricing e aeroportos tornam essas corridas imprevisíveis
- Neural Network: melhor R² (0.815), empate em RMSE com DT, melhor MAE ($4.94)
- Decision Tree: equivalente em RMSE, mais robusto e interpretável (importance plot direto)
- LR: mais rápido e interpretável, R² competitivo, bom baseline

---

## Slide 12 — Desafios técnicos enfrentados

Esta seção é crítica para o requisito "desafios enfrentados". Escolher 4–5 dos mais interessantes:

**D1 — Schema incompatível entre documentação e dataset real**
- O CLAUDE.md documentava o schema TLC bruto (24 colunas: `hvfhs_license_num`, `base_passenger_fare`, `trip_miles`...)
- O arquivo real é o dataset Kaggle preprocessado (15 colunas: `business`, `passenger_fare`, `trip_length`...)
- Descoberto ao rodar o EDA pela primeira vez. Exigiu refatoração completa de todos os notebooks e `_lib.py`

**D2 — Incompatibilidade de versão Spark (3.5.0 vs 3.5.8)**
- Workers rodavam `apache/spark:3.5.8-python3`, driver usava JARs 3.5.0 instalados na imagem base
- Erro: `InvalidClassException: org.apache.spark.scheduler.Task` ao submeter qualquer job
- Correção: `ENV SPARK_HOME=/opt/conda/lib/python3.11/site-packages/pyspark` no Dockerfile

**D3 — VectorIndexer com valores não vistos no treino**
- `month_of_year` tem valores 3,4,5 no treino e 6,7,8 no teste (split temporal)
- Sem `handleInvalid='keep'`, o transform falhava com `Invalid value 8.0`
- Não é bug de código — é a consequência direta do split temporal correto

**D4 — Viés temporal na amostragem da NN**
- `RAND(seed) + LIMIT 2M` preenchia a cota com March inteiro (~1.8M) + início de April
- Modelo treinado sem nenhuma linha de May → RMSE $11.62 (pior que 1.6M)
- Correção: UNION ALL com filtros explícitos por mês garante representação igual de Mar/Abr/Mai

**D5 — OOM em toPandas() com 5M linhas**
- Kernel Python morreu ao tentar trazer 5M linhas para o processo PyTorch
- Causa: driver JVM (4GB) + coleta de dados (~1GB) + DataFrame pandas (~800MB) ultrapassa 8GB da máquina
- Lição: em produção, usaria Horovod ou Spark-PyTorch para treino distribuído

---

## Slide 13 — Sugestões para melhorias futuras

**Dados:**
- Expandir para o recorte completo Jan/2022–Ago/2023 com hardware adequado (16GB+)
- Incluir dados externos: clima (NYC Open Data), eventos especiais, greves de transporte público
- Enriquecer zonas TLC com coordenadas geográficas reais para features de distância geodésica

**Modelos:**
- **Gradient Boosted Trees** (XGBoost/LightGBM): tipicamente supera Decision Tree em regressão tabular sem restrição de enunciado
- **Treino distribuído da NN**: Horovod + Spark para usar os 110M de treino em vez de 1.6M
- **Ensemble**: combinar previsões de LR + DT + NN com meta-learner

**Engenharia de features:**
- Distância real entre zonas via API (ao invés de apenas `same_borough`)
- Taxa histórica de surge pricing por zona e hora (requer dados adicionais)
- Sequências temporais de demanda por zona (LSTM/Transformer)

**Infraestrutura:**
- MLflow para rastreamento de experimentos (substitui `model_comparison.csv` manual)
- Delta Lake para versionamento do dataset preparado em disco
- Testes de data quality automáticos com Great Expectations

---

## Slide 14 — Conclusão

**O que mostrar:**

> Com PySpark, Spark SQL e MLlib, processamos **110 milhões de corridas** de Uber e Lyft de Nova York em um cluster Spark local containerizado, treinando 3 modelos de regressão para prever `passenger_fare`.

**Resultados-chave:**
- Neural Network explica **81.5% da variação** da tarifa (R²=0.815) com MAE de **$4.94**
- Decision Tree fica em 81.3% (R²=0.813) com MAE de **$5.13** sobre o conjunto completo
- Em uma tarifa mediana de ~$22, MAE de $4.94 representa ~22% de erro relativo

**O que o projeto demonstrou:**
1. Pipeline Big Data end-to-end reproduzível em Docker
2. Spark SQL para ETL/EDA em escala de 100M+ linhas
3. Split temporal correto evita data leakage em séries temporais
4. Trade-offs reais entre modelos distribuídos (LR/DT) e locais (NN)
5. Diagnóstico e correção de bugs de infraestrutura (versão JAR, permissões, OOM)

---

## Demonstração prática (se solicitado pela banca)

Preparar para mostrar ao vivo ou em vídeo gravado:

1. **Spark UI** (`localhost:8080`) — cluster com 2 workers, mostrar jobs executados
2. **Jupyter Lab** (`localhost:8888`) — abrir notebook 04 e mostrar uma célula de avaliação já executada
3. **model_comparison.csv** — mostrar o histórico de runs com métricas
4. **Residual plot** — `results/residual_analysis_neural_network.png`

---

## Notas de apresentação

- Não é necessário mostrar todo o código nos slides — mostrar outputs (gráficos, tabelas de métricas)
- Ao falar de R²: "0.815 significa que o modelo explica 81.5% da variação na tarifa. Os 18.5% restantes são surge pricing não modelado, condições de tráfego em tempo real e fatores que não estão no dataset."
- Ao falar de MAE $4.94: "Em uma corrida típica de $22, o modelo erra em média $4.94 — um erro relativo de 22%. Para contexto: a própria plataforma aplica surge multipliers que podem variar a tarifa em 30–50% em minutos."
- Ao falar de desafios: mostrar confiança — bugs complexos resolvidos demonstram maturidade técnica
