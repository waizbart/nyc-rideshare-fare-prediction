# NYC Rideshare Fare Prediction

Projeto final da disciplina de Big Data. Pipeline de regressão sobre tarifa de corridas de aplicativo em Nova York, usando PySpark sobre o dataset HVFHV da NYC Taxi & Limousine Commission.

Este arquivo é o contexto persistente para o Claude Code. Leia inteiro antes de propor alterações.

---

## 1. Objetivo

Construir, entregar e apresentar um projeto end-to-end de Big Data envolvendo:

1. Ambiente distribuído containerizado (Spark cluster via Docker)
2. EDA + preprocessing sobre dataset > 1 GB
3. Três modelos preditivos para o mesmo target

Tarefa: **regressão**. Target: `base_passenger_fare`.

---

## 2. Dataset

### Fonte

NYC TLC HVFHV (High-Volume For-Hire Vehicle) Trip Records, redistribuído no Kaggle como "NYC Rideshare Raw Data" (aaronweymouth). Fonte primária equivalente: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### Características

- Período do dataset bruto da TLC: janeiro/2022 a agosto/2023
- **Recorte em uso no projeto**: 2023-03 a 2023-08 (6 meses, ~3 GB compactados). Cabe na RAM dos workers e mantém o `SPLIT_DATE=2023-06-01` viável (3 meses de treino, 3 meses de teste).
- Granularidade: uma linha por viagem
- Formato: Parquet
- Volume bruto referência: ~360 milhões de linhas, ~12 a 15 GB compactados
- Cobre 4 operadoras: Uber, Lyft, Via, Juno

### Esquema (24 colunas)

| Coluna | Tipo | Descrição |
|---|---|---|
| `hvfhs_license_num` | string | HV0002 Juno, HV0003 Uber, HV0004 Via, HV0005 Lyft |
| `dispatching_base_num` | string | Base que despachou a corrida |
| `originating_base_num` | string | Base que recebeu a solicitação |
| `request_datetime` | timestamp | Solicitação do passageiro |
| `on_scene_datetime` | timestamp | Chegada do motorista (cobertura desigual entre operadoras) |
| `pickup_datetime` | timestamp | Início da corrida |
| `dropoff_datetime` | timestamp | Fim da corrida |
| `PULocationID` | int | Zona TLC de embarque |
| `DOLocationID` | int | Zona TLC de desembarque |
| `trip_miles` | float | Distância em milhas |
| `trip_time` | int | Duração em segundos |
| `base_passenger_fare` | float | **TARGET**, tarifa base antes de tolls/tips/taxes |
| `tolls` | float | Pedágios |
| `bcf` | float | Black Car Fund |
| `sales_tax` | float | Imposto estadual |
| `congestion_surcharge` | float | Taxa de congestionamento (vai para a MTA) |
| `airport_fee` | float | Taxa de aeroporto |
| `tips` | float | Gorjeta |
| `driver_pay` | float | Pagamento ao motorista |
| `shared_request_flag` | string Y/N | Passageiro aceitou compartilhada |
| `shared_match_flag` | string Y/N | Houve pareamento real |
| `access_a_ride_flag` | string Y/N | Corrida via MTA Access-A-Ride |
| `wav_request_flag` | string Y/N | Solicitou veículo acessível |
| `wav_match_flag` | string Y/N | Corrida em veículo acessível |

### Limitações conhecidas (críticas, documentar no relatório)

1. **Sem ID de motorista ou passageiro**. Não dá para calcular receita por motorista, viagens por dia, churn.
2. **`shared_match_flag` tem semântica diferente entre Lyft e Uber**: para Lyft, o flag pode indicar tanto pareamento real quanto solicitação sem match. Comparações Uber vs Lyft sobre pool são enviesadas sem ajuste.
3. **`PULocationID/DOLocationID` 264 e 265 = "Unknown" e "Outside NYC"**. Devem ser filtrados.
4. **Sem latitude/longitude**, só zonas. Análise geoespacial só agregada.
5. **`airport_fee` só passou a ser preenchido em meados de 2022**.
6. **`on_scene_datetime` tem cobertura desigual** entre operadoras, especialmente Uber. Não usar como feature crítica.
7. **Viés de sobrevivência**: só corridas concluídas. Cancelamentos não estão no dataset.
8. **Desbalanceamento entre operadoras**: Uber representa ~70% das linhas. Comparações diretas precisam ponderar.

---

## 3. Decisões de modelagem (já tomadas, não revisitar sem motivo forte)

### Target = `base_passenger_fare`, não `driver_pay`

**Justificativa**: `driver_pay` é regulatorialmente determinado pela NYC TLC Local Law 150/2018 (regra de pagamento mínimo, vigente desde fevereiro/2019). A fórmula essencial é `pay = (miles * per_mile_rate + minutes * per_minute_rate) / utilization_factor`. Um regressor sobre `driver_pay` recupera essa fórmula com R² tipicamente acima de 0,97, o que torna o problema trivial e analiticamente menos defensável. `base_passenger_fare` mantém sinal de mercado real (surge pricing, variação por zona, demanda).

### Split temporal, não aleatório

`SPLIT_DATE = "2023-06-01"`. Treino antes, teste depois. `randomSplit` em série temporal vaza informação futura no treino e infla métricas.

### Filtros de cleaning baseados em percentis, não cutoffs fixos

Distribuições de tarifa, distância e velocidade são heavy-tailed. Filtros fixos arbitrários (ex: "velocidade entre 1 e 80 mph") são mal calibrados para NYC, onde a velocidade média em Manhattan no horário de pico está em torno de 7 mph. Usar percentis 0,1% e 99,9% via `PERCENTILE_APPROX`.

### Margem de plataforma: NÃO calcular como `gross_fare - driver_pay`

`gross_fare` inclui `sales_tax`, `congestion_surcharge`, `bcf`, `airport_fee`, que não são receita da Uber/Lyft (vão para estado, MTA, fundo regulatório). Estimativa correta de "take" seria apenas `base_passenger_fare - driver_pay`, e mesmo assim é só estimativa porque faltam custos operacionais, incentivos, promoções, seguros.

---

## 4. Stack técnica

### Travada (não mudar)

- **Python 3.10+, PySpark 3.5**
- **Docker Compose** com Spark master + 2 workers + Jupyter
- **Sem HDFS, sem MinIO, sem Hadoop**. Volume local montado nos containers.
- **Spark SQL via `spark.sql()`** para todo ETL e EDA. NÃO usar DataFrame API (`df.filter()`, `df.groupBy()`, etc.) para data prep. ML pipeline em Python é OK.
- **Spark MLlib** para Linear Regression e Decision Tree
- **PyTorch** para Rede Neural

### Por que essas restrições

- Sem Hadoop/MinIO: simplicidade de setup, restrição da equipe
- Spark SQL puro: requisito do enunciado, "precisa ser aquele tipo SQL"
- PyTorch para NN: MLlib não tem regressor de rede neural (`MultilayerPerceptronClassifier` é só para classificação)

---

## 5. Arquitetura do ambiente

```
docker-compose:
  spark-master       (Spark UI: localhost:8080)
  spark-worker-1     (4G RAM, 2 cores)
  spark-worker-2     (4G RAM, 2 cores)
  jupyter            (Jupyter Lab: localhost:8888)
```

Volume `./data:/data` montado em todos os 4 containers. Como não há storage distribuído, os workers precisam enxergar o mesmo path do driver.

Dependências adicionais a instalar no Jupyter: `torch` (não vem na imagem `jupyter/pyspark-notebook`).

---

## 6. Estrutura do repositório

```
nyc-rideshare-fare-prediction/
├── CLAUDE.md                         # este arquivo
├── README.md                         # como rodar
├── docker-compose.yml
├── Dockerfile.jupyter                # imagem custom com PyTorch
├── data/                             # gitignore, Parquet + lookup
│   ├── fhvhv_tripdata_*.parquet
│   └── taxi_zone_lookup.csv
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA em Spark SQL
│   ├── 02_preprocessing.ipynb        # camada silver em SQL
│   ├── 03_linear_regression.ipynb    # baseline MLlib
│   ├── 04_decision_tree.ipynb        # MLlib
│   └── 05_neural_network.ipynb       # PyTorch
├── models/                           # gitignore
├── results/
│   ├── model_comparison.csv
│   ├── residual_analysis.png
│   └── feature_importance.png
├── slides/
│   └── apresentacao_final.pdf
└── docs/
    ├── PLAN.md                       # timeline detalhado
    ├── data_dictionary.md
    └── decisions_log.md              # log de decisões durante o projeto
```

---

## 7. Convenções de código

### Spark SQL

- Sempre usar `CREATE OR REPLACE TEMPORARY VIEW` para etapas intermediárias
- CTEs (`WITH ... AS`) para queries complexas, ao invés de subqueries aninhadas
- Nomes de view em snake_case com prefixo de camada: `trips_bronze`, `trips_silver`, `trips_features`
- Documentar a justificativa de cada filtro inline com comentário SQL

Exemplo do padrão esperado:

```python
spark.sql("""
    CREATE OR REPLACE TEMPORARY VIEW trips_silver AS
    WITH bounds AS (
        SELECT
            PERCENTILE_APPROX(base_passenger_fare, 0.001) AS fare_low,
            PERCENTILE_APPROX(base_passenger_fare, 0.999) AS fare_high
        FROM trips_bronze
        WHERE base_passenger_fare > 0
    )
    SELECT t.*
    FROM trips_bronze t
    CROSS JOIN bounds b
    WHERE t.pickup_datetime  >= t.request_datetime    -- temporal
      AND t.dropoff_datetime >  t.pickup_datetime
      AND t.trip_miles BETWEEN 0.1 AND 100             -- distância
      AND t.trip_time  BETWEEN 60  AND 14400           -- duração 1min a 4h
      AND t.base_passenger_fare BETWEEN b.fare_low AND b.fare_high
      AND t.PULocationID NOT IN (264, 265)             -- zonas inválidas
      AND t.DOLocationID NOT IN (264, 265)
""")
```

### Python ML pipeline

- Type hints em funções públicas
- Constantes em UPPER_SNAKE no topo do módulo
- Seeds fixos em todo lugar que tem aleatoriedade (`seed=42`)
- Salvar modelos em `models/` com nome descritivo

### Notebooks

- Primeira célula sempre cria a `SparkSession` com `master("spark://spark-master:7077")`
- Última célula sempre tem `spark.stop()`
- Plots em pandas (após `.toPandas()` em agregados pequenos), não plotar diretamente do Spark

---

## 8. Features engineered

A partir da camada silver, gerar:

**Temporais**
- `pickup_hour` = `HOUR(pickup_datetime)`
- `pickup_dow` = `DAYOFWEEK(pickup_datetime)`
- `pickup_month` = `MONTH(pickup_datetime)`
- `is_weekend` = `pickup_dow IN (1,7)`
- `is_rush_hour` = `pickup_hour BETWEEN 7 AND 9 OR pickup_hour BETWEEN 16 AND 19`
- `is_late_night` = `pickup_hour >= 22 OR pickup_hour <= 4`

**Espaciais** (após join com taxi_zone_lookup)
- `pu_borough`, `do_borough`
- `same_borough` = boolean
- `pickup_airport` = `PULocationID IN (1, 132, 138)` (EWR, JFK, LGA)
- `dropoff_airport` = idem para DOLocationID

**Operacionais**
- `wait_time_sec` = `pickup_datetime - request_datetime`
- `speed_mph` = `trip_miles / (trip_time / 3600.0)` (filtrar por percentis)

**Flags convertidas**
- `shared_req` = `shared_request_flag = 'Y'` -> int
- `wav_req` = `wav_request_flag = 'Y'` -> int

**Categóricas para encoding posterior**
- `hvfhs_license_num`
- `pu_borough`, `do_borough`

---

## 9. Modelos

### Modelo 1: Linear Regression (MLlib)

Baseline. `pyspark.ml.regression.LinearRegression`. Sem regularização inicial, depois experimentar `regParam` e `elasticNetParam`.

### Modelo 2: Decision Tree Regressor (MLlib)

`pyspark.ml.regression.DecisionTreeRegressor`. **Atenção**: o enunciado pede árvore, não boosting nem ensemble. NÃO trocar por `GBTRegressor` ou `RandomForestRegressor` mesmo se o resultado for melhor. `maxDepth` máximo 10.

### Modelo 3: Rede Neural (PyTorch)

Arquitetura sugerida:
- Input: features numéricas + one-hot das categóricas
- Hidden 1: 128, ReLU, Dropout 0.2
- Hidden 2: 64, ReLU, Dropout 0.2
- Hidden 3: 32, ReLU
- Output: 1 (regressão)
- Loss: MSE, Optimizer: Adam (lr=1e-3)
- Early stopping em validation set
- StandardScaler nas features numéricas (crítico, sem isso não converge bem)

Estratégia de dados: amostra de 1-2M linhas para treino. PyTorch puro, não distribuído. É defensável: o foco da NN no projeto é demonstrar capacidade do modelo, não throughput.

### Métricas obrigatórias para os 3 modelos

- RMSE (USD)
- MAE (USD)
- R²
- Tempo de treino
- Erro estratificado por borough
- Erro estratificado por bucket de distância (0-2 mi, 2-5 mi, 5-10 mi, 10+ mi)
- Análise de resíduos (histograma + scatter resíduo vs predito)

### Expectativa de performance

Com as features atuais sobre `base_passenger_fare`:

- Linear Regression: R² ~0.85, RMSE ~$5
- Decision Tree (depth 8-10): R² ~0.90, RMSE ~$4
- Neural Network: R² ~0.91-0.93, RMSE ~$3.50

Se algum modelo bater R² > 0.99: vazamento, provavelmente alguma feature derivada do target ficou no input.
Se ficar abaixo de R² ~0.80: problema de cleaning ou escala das features.

---

## 10. Timeline

Apresentação entre 11/05 e 27/05/2026. Plano de 4 semanas (ajustar se a data for antes):

- **Semana 1 (28/04 a 04/05)**: Docker up, ingestão, EDA inicial
- **Semana 2 (05/05 a 11/05)**: Preprocessing silver, Linear Regression, Decision Tree
- **Semana 3 (12/05 a 18/05)**: Neural Network, tuning, métricas finais
- **Semana 4 (19/05 a 27/05)**: Análise de resíduos, slides, demo, apresentação

Detalhamento dia a dia em `docs/PLAN.md`.

---

## 11. O que NÃO fazer

- Não usar DataFrame API para ETL. Spark SQL puro.
- Não usar HDFS, MinIO ou qualquer storage distribuído além do volume local.
- Não trocar `DecisionTreeRegressor` por GBT ou Random Forest.
- Não usar `randomSplit` para train/test.
- Não calcular margem da plataforma como `gross_fare - driver_pay`.
- Não fazer comparação direta de "taxa de pool" entre Uber e Lyft sem ressalvar a diferença de semântica do `shared_match_flag`.
- Não incluir zonas 264 e 265 nas análises geográficas.
- Não usar `on_scene_datetime` como feature em modelo (cobertura desigual).
- Não tratar `driver_pay` como sinal de mercado (é regulatório).

---

## 12. Comandos úteis

### Subir o ambiente

```bash
mkdir -p data notebooks models results docs
docker compose up -d
docker compose ps
```

### Validar cluster

```bash
# Spark UI em localhost:8080 deve mostrar 2 workers ALIVE
# Jupyter em localhost:8888 sem token
```

### Smoke test em PySpark

```python
from pyspark.sql import SparkSession

spark = (SparkSession.builder
    .appName("smoke")
    .master("spark://spark-master:7077")
    .config("spark.executor.memory", "3g")
    .getOrCreate())

spark.read.parquet("/data/fhvhv_tripdata_*.parquet").createOrReplaceTempView("trips")
spark.sql("SELECT hvfhs_license_num, COUNT(*) FROM trips GROUP BY 1").show()
```

Esperado: 4 linhas (HV0002, HV0003, HV0004, HV0005).

### Derrubar o ambiente

```bash
docker compose down
docker compose down -v  # também remove volumes anônimos
```

---

## 13. Observações para o Claude Code

- Quando criar notebooks novos, seguir o template das primeira/última células descrito na seção 7.
- Sempre que propor um filtro de cleaning, calcular e reportar o percentual de linhas removidas.
- Sempre que propor uma feature, justificar por que ela é informativa para `base_passenger_fare`.
- Antes de rodar treino full, sempre testar em sample (`TABLESAMPLE` em SQL ou `.sample()` no DF que vai para o ML pipeline).
- Critical: este é um trabalho acadêmico em equipe. Mudanças significativas em decisões de modelagem (mudar target, mudar split, mudar stack) devem ser sinalizadas e justificadas, não feitas silenciosamente.
- Resposta deve ser crítica, com base em dados/fontes, não confirmar acriticamente. Apontar erros de raciocínio do usuário quando existirem.
