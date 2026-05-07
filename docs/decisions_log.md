# Decisions Log

## 2026-04-28

### Decisao: target oficial

- `base_passenger_fare`
- Motivo: `driver_pay` e fortemente regulado e transforma a regressao em um problema quase trivial.

### Decisao: split de treino e teste

- `SPLIT_DATE = "2023-06-01"`
- Motivo: evita vazamento temporal e segue a premissa do projeto.

### Decisao: stack

- Spark SQL para EDA e preprocessing
- Spark MLlib para Linear Regression e Decision Tree
- PyTorch para Neural Network
- Docker Compose com master, dois workers e Jupyter

### Decisao: estrategia de iteracao

- Bootstrap inicial com `fhvhv_tripdata_2023-08.parquet`
- Expansao parcial para `2023-03` a `2023-08` (6 meses, ~3 GB compactado), nao para o range 2022-2023 completo
- Motivo: reduzir custo de iteracao e risco de OOM nos workers de 4 GB; com 6 meses cabe na RAM, o split `2023-06-01` mantem 3 meses de cada lado e o pipeline roda end-to-end sem amostragem

### Restricao operacional descoberta no bootstrap

- `2023-08` sozinho nao permite treinar os modelos com `SPLIT_DATE = "2023-06-01"`
- Motivo: todo o dataset cairia no lado de teste; para modelagem e preciso ter observacoes antes e depois da data de corte

### Decisao: reproducibilidade

- Seeds fixos em notebooks e pipelines
- Dependencias principais concentradas em `Dockerfile.jupyter`
- Saidas versionaveis limitadas a codigo, docs e metricas tabulares

### Pendente

- Confirmar data final da apresentacao para validar se o cronograma de 4 semanas permanece realista
- Confirmar se o repositorio sera mantido privado ate a entrega

## 2026-04-28 - revisao critica do pipeline

Revisao linha-a-linha dos notebooks 02-05 antes da primeira rodada end-to-end. Bugs e violacoes identificados e corrigidos in-place:

### B1 - Decision Tree tratava categoricas como continuas

- Local: `notebooks/04_decision_tree.ipynb`, cell do pipeline
- Sintoma: `StringIndexer` produz indices 0,1,2,... que entram no `VectorAssembler` sem metadata categorico. O `DecisionTreeRegressor` faz splits ordinais (ex: `pu_borough_idx <= 1.5`), impondo ordem artificial entre boroughs e degradando o modelo.
- Correcao: `VectorIndexer(maxCategories=8)` entre o assembler e o DT. Cobre `hvfhs_license_num` (4), `pu/do_borough` (6 + Unknown) e marca tambem as binarias - efeito benigno.

### B2 - `same_borough` silenciava NULL

- Local: `notebooks/02_preprocessing.ipynb`, cell silver
- Sintoma: `LEFT JOIN` em `taxi_zone_lookup` pode retornar NULL para zonas nao mapeadas. `NULL = NULL` e NULL em SQL, caia no ELSE e marcava `0`, fingindo que sao boroughs diferentes.
- Correcao: `CASE WHEN pu_borough IS NULL OR do_borough IS NULL THEN 0 WHEN pu_borough = do_borough THEN 1 ELSE 0 END`. Conservador (zona desconhecida nao confunde modelos), e o audit do cell anterior agora reporta `null_borough_rows`.

### B3 - Train/test split em DataFrame API violava CLAUDE.md

- Local: `notebooks/03_linear_regression.ipynb`, `04_decision_tree.ipynb`, `05_neural_network.ipynb` na cell de split
- Sintoma: split com `silver_df.filter(F.col('pickup_datetime') < F.lit(SPLIT_DATE))`. CLAUDE.md exige Spark SQL puro para data prep, e o split E data prep.
- Correcao: criar `TEMPORARY VIEW trips_silver` e usar `spark.sql(f"SELECT * FROM trips_silver WHERE pickup_datetime < TIMESTAMP '{SPLIT_DATE}'")`. No notebook 05 a amostragem tambem foi migrada para SQL via `RAND(seed) < fraction LIMIT n`.

### B4 - sklearn `mean_squared_error(squared=False)` removido em sklearn 1.6

- Local: `notebooks/05_neural_network.ipynb`, cell de avaliacao
- Sintoma: parametro `squared` foi deprecado em sklearn 1.4 e removido em 1.6, e a imagem base traz uma versao moderna.
- Correcao: `np.sqrt(mean_squared_error(...))`. Mais robusto e nao depende de versao.

### B5 - val split aleatorio na NN

- Local: `notebooks/05_neural_network.ipynb`, cell de get_dummies
- Decisao: manter `train_test_split(train_pdf, test_size=0.2)` para val. Nao e bug: o test set ja foi separado temporalmente. Comentario explicito adicionado para evitar confusao em revisao.

### B6 - alias SQL `do` confunde leitura

- Local: `notebooks/02_preprocessing.ipynb`, cell silver
- Sintoma: `LEFT JOIN taxi_zone_lookup do` funciona no Spark mas e fragil/ilegivel.
- Correcao: renomeado para `do_zone` (e `pu` -> `pu_zone` por simetria).

### Audit ampliado

`02_preprocessing` cell de audit agora tambem reporta:
- `null_borough_rows` (suporte ao B2)
- `null_airport_fee_rows` (rastreabilidade do `airport_fee` ausente em corridas pre-meados/2022, ja documentado em CLAUDE.md secao 2)

## 2026-04-29 - rodada 2 de melhorias (perspectiva de engenharia de dados senior)

Antes do primeiro end-to-end, segunda rodada de revisao focada em qualidade de codigo, reprodutibilidade e arquitetura. Mudancas aplicadas:

### M1 - Deps Python pinadas no `Dockerfile.jupyter`

- Sintoma: somente `pyspark==3.5.0` estava pinado. `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` flutuavam. O bug B4 (sklearn>=1.6 removendo `squared=False`) foi consequencia direta disso.
- Correcao: pinar `pandas==2.2.3`, `numpy==1.26.4`, `pyarrow==16.1.0`, `scikit-learn==1.5.2`, `matplotlib==3.9.2`, `seaborn==0.13.2`, `torch==2.4.1` + companheiros. Imagem agora reconstroi identica entre maquinas.
- Tambem instalado `git` no container para habilitar registro de `git_sha` em `model_comparison.csv`.

### M2 - Modulo `notebooks/_lib.py` para constantes e factory de SparkSession

- Sintoma: `NUMERIC_COLS`, `CATEGORICAL_COLS`, `SEED`, `SPLIT_DATE`, `SILVER_PATH`, `TARGET_COL` e o builder do SparkSession estavam copiados em 03/04/05. Risco de drift silencioso entre modelos (alterar features em um sem propagar nos outros distorce a comparacao).
- Correcao: extracao para `_lib.py`. Notebooks importam via `sys.path.insert(0, '/home/jovyan/work/notebooks')` e `from _lib import ...`. Single source of truth.

### M3 - SparkSession com `shuffle.partitions=32` e AQE explicito

- Sintoma: `spark.sql.shuffle.partitions=200` para ~3 GB de dados gerava tasks de <1s onde o overhead de scheduler dominava. AQE habilitado por default no Spark 3.5 mas nao explicitado.
- Correcao: `shuffle.partitions=32`, `spark.sql.adaptive.enabled=true`, `spark.sql.adaptive.coalescePartitions.enabled=true`. Documenta intencao e protege de regressao em upgrades.

### M4 - Persistencia de `cleaning_bounds`

- Sintoma: `PERCENTILE_APPROX` recomputado a cada run, sem registro do corte usado. Reruns produzem silvers ligeiramente diferentes; auditoria fica impossivel.
- Correcao: bounds gravados em `/results/cleaning_bounds.json` no notebook 02 logo apos o calculo. Material para o relatorio e base para detectar drift.

### M5 - Piso de `speed_mph > 0.5`

- Sintoma: o filtro percentilico admite `speed_mph = 0` se houver muitas viagens com `trip_miles ~ 0`. Corridas a 0 mph sao ruido (cancelamento mascarado, GPS travado).
- Correcao: `AND speed_mph > 0.5` adicionado ao silver. Percentual removido a mais e auditado pelo gate da M7.

### M6 - `repartition('pickup_year_month')` antes do write da silver

- Sintoma: `partitionBy('pickup_year_month')` sem repartition previo gera ate `shuffle.partitions x particoes_logicas` part-files. Com 32 x 6 = ate 192 arquivos pequenos.
- Correcao: `repartition('pickup_year_month')` antes do write -> 1 arquivo Parquet por mes (~6 arquivos totais).

### M7 - DQ gate explicito no silver

- Sintoma: a auditoria reportava drop% mas nao falhava o pipeline. Silver corrompida silenciosamente seguia para os modelos.
- Correcao: `assert pct_rows_removed < 20.0` apos a contagem. Schema drift, bounds mal calibrados ou bronze suja agora abortam o notebook.

### M8 - Checkpoint completo da NN

- Sintoma: `torch.save` no notebook 05 persistia `state_dict` e `feature_columns` mas nao `scaler.mean_` / `scaler.scale_`. Inferencia em outra maquina exigia retreinar o scaler.
- Correcao: checkpoint inclui `scaler_mean`, `scaler_scale`, `categorical_cols` e `numeric_cols`. Train/serve parity restaurada.

### M9 - `model_comparison.csv` append-only com `run_id`/`timestamp_utc`/`git_sha`

- Sintoma: o codigo anterior fazia upsert por `model`, sobrescrevendo runs anteriores. Timeline de experimentos era perdida.
- Correcao: helper `_lib.append_metric` adiciona uma linha por run com identificador, timestamp UTC e git SHA curto. Permite comparar runs ao longo do tempo e correlacionar metrica com versao do codigo.

## 2026-05-06 - auditoria completa pre-execucao e adaptacao ao schema real

### B7 - DATA_GLOB incompativel com o arquivo real

- Local: `notebooks/_lib.py`
- Sintoma: DATA_GLOB = '/data/fhvhv_tripdata_*.parquet' nao casa com o arquivo real 'rideshare_data.parquet'. Todos os notebooks falhavam ao tentar carregar os dados.
- Correcao: DATA_GLOB = '/data/rideshare_data.parquet'

### B8 - Schema do dataset diferente do previsto no CLAUDE.md

- Contexto: o arquivo 'rideshare_data.parquet' (12 GB, 365M linhas) e o dataset Kaggle "NYC Rideshare Raw Data" ja preprocessado, nao o HVFHV bruto da TLC. O schema e completamente diferente:
  - Target: 'passenger_fare' (era 'base_passenger_fare')
  - Colunas de identificacao: 'business' (Uber/Lyft), nao 'hvfhs_license_num' (HV0002-HV0005)
  - Sem timestamps exatos: apenas 'date' (date), 'hour_of_day' (int)
  - Features temporais ja computadas: 'month_of_year', 'week_of_year', 'time_of_day'
  - Wait time: 'request_to_pickup' (segundos) em vez de calcular unix_timestamp
  - Distancia: 'trip_length' (milhas) em vez de 'trip_miles'
  - Duracao: 'total_ride_time' (segundos) em vez de 'trip_time'
  - Sem flags: sem shared_request_flag, wav_request_flag, airport_fee, congestion_surcharge, bcf
  - Colunas com LEAKAGE do target: 'hourly_rate' e 'dollars_per_mile' derivadas de 'passenger_fare'
  - Sem Juno (HV0002) nem Via (HV0004), apenas Uber e Lyft
- Correcao: refatoracao completa de todos os notebooks e _lib.py para usar o schema real.
- Colunas excluidas da silver por leakage: 'hourly_rate', 'dollars_per_mile' (derivadas do target), 'rideshare_profit' (derivada do target), 'driver_total_pay' (regulatorio, nao deve ser feature), 'on_scene_to_pickup', 'on_scene_to_dropoff' (cobertura desigual).

### B9 - SPARK_HOME apontando para JARs 3.5.0 enquanto workers rodam 3.5.8

- Sintoma: 'InvalidClassException: org.apache.spark.scheduler.Task' ao submeter tarefas. Worker rejeita tasks serializadas pelo driver.
- Causa: imagem base 'quay.io/jupyter/pyspark-notebook:spark-3.5.0' tem SPARK_HOME=/usr/local/spark com JARs 3.5.0. Instalar pyspark==3.5.8 via pip nao sobrescreve SPARK_HOME; o driver continua usando JARs 3.5.0.
- Correcao: adicionar 'ENV SPARK_HOME=/opt/conda/lib/python3.11/site-packages/pyspark' no Dockerfile.jupyter.

### B10 - Filtro de escopo temporal ausente no bronze

- Sintoma: sem filtro, notebooks carregariam todos os 365M linhas (Dez/2021-Ago/2023), causando OOM nos workers (4G cada) e violando o recorte de 6 meses definido em CLAUDE.md.
- Correcao: adicionar 'WHERE date >= 2023-03-01 AND date < 2023-09-01' ao criar trips_bronze em 01_eda e 02_preprocessing.

### B11 - driver_pay presente no SELECT de trips_silver

- Sintoma: 'driver_total_pay' (equivalente do driver_pay) era selecionado na view trips_silver, violando a regra do CLAUDE.md de nao usar como feature.
- Correcao: removido do SELECT e nao incluido em NUMERIC_COLS.

### B12 - Falta de analise de residuos no Decision Tree

- Sintoma: CLAUDE.md exige analise de residuos (histograma + scatter) para todos os 3 modelos. Notebook 04 so tinha feature importance.
- Correcao: celula de residuos adicionada apos feature importance, seguindo o padrao do LR.

### Atualizado: SPLIT_DATE mantido em 2023-06-01

- O dataset contem dados de Mar a Ago 2023 no recorte definido: 3 meses de treino (Mar-Mai) e 3 meses de teste (Jun-Ago). Distribuicao adequada para series temporais.
- Confirmado: ~116.2M linhas no recorte de 6 meses (bronze), 110.1M apos DQ gate (silver, drop 5.26%).

## 2026-05-07 - execucao end-to-end e otimizacao da NN

### Execucao completa do pipeline

Todos os 5 notebooks executados em sequencia via `jupyter nbconvert --execute`. Resultados finais:

| Modelo | R² | RMSE | MAE | Treino |
|--------|----|------|-----|--------|
| Linear Regression | 0.806 | $9.07 | $5.76 | 241s |
| Decision Tree (depth 8) | 0.813 | $8.91 | $5.13 | 462s |
| Neural Network | 0.796 | $9.13 | $4.72 | 821s |

### B13 - Stale kernel causava re-uso de estado entre execucoes nbconvert

- Sintoma: `jupyter nbconvert --execute --inplace` conectava a um kernel Jupyter ja rodando que tinha `train_pdf` de uma run anterior em memoria. Celulas 1 e 2 (imports + toPandas) ficavam com `execution_count=None` e as celulas seguintes usavam dados da run anterior sem re-executar.
- Diagnostico: `execution_count=None` em celulas executadas indica que o kernel nao foi limpo antes da execucao. Celulas 3-9 com contadores sequenciais confirmavam re-uso de estado.
- Correcao: `docker compose restart jupyter` antes de cada execucao nbconvert critica, ou adicionar `--ExecutePreprocessor.kernel_name=python3` para forccar novo kernel.

### B14 - OOM no kernel ao tentar toPandas() com 5M linhas

- Sintoma: kernel Python morreu (`DeadKernelError`) ao tentar `toPandas()` com `MAX_TRAIN_ROWS = 5_000_000` em maquina de 8GB.
- Causa: driver JVM (4G configurado) + coleta de 5M linhas (~1GB de dados + overhead) + DataFrame pandas (~800MB) + workers em execucao. Pico estimado de 9-10GB em maquina de 7.6GB efetivos.
- Correcao: reduzir para `executor_memory='2g', driver_memory='3g'` no `build_spark()` da NN, e `MAX_TRAIN_ROWS <= 2_500_000`.

### D1 - Decisao: limite seguro de amostra da NN em 8GB de RAM

- Maximo testado sem OOM: 2.5M linhas com `executor_memory='2g', driver_memory='3g'`.
- Maximo com `executor_memory='3g', driver_memory='4g'` (padrao): 1.6M linhas.
- Justificativa: PyTorch nao e distribuido; o custo de `toPandas()` e linear no numero de linhas e concentrado no driver. Aumentar alem de 2.5M exige mais RAM ou reducao das memorias dos workers.

### D2 - Decisao: amostragem estratificada por mes para a NN

- Contexto: `RAND(seed) < fraction LIMIT N` com silver particionado por mes escaneia particoes em ordem, preenchendo a cota prioritariamente com os primeiros meses (Marco >> Abril >> Maio). Com `LIMIT 2M`, o conjunto de treino era ~1.8M de Marco + ~200k de Abril, sem nenhuma linha de Maio.
- Efeito observado: run com 2M linhas biased obteve RMSE $11.62 (pior que 1.6M, RMSE $9.13). O modelo treinado em Marco/Abril generalizava mal para Junho-Agosto.
- Correcao adotada: UNION ALL de 3 subqueries com filtro de data explicito por mes e LIMIT por particao:

```sql
(SELECT ... WHERE date >= '2023-03-01' AND date < '2023-04-01' AND RAND(42) < 0.10 LIMIT 533333)
UNION ALL
(SELECT ... WHERE date >= '2023-04-01' AND date < '2023-05-01' AND RAND(42) < 0.10 LIMIT 533333)
UNION ALL
(SELECT ... WHERE date >= '2023-05-01' AND date < '2023-06-01' AND RAND(42) < 0.10 LIMIT 533333)
```

- Resultado: estratificado 1.3M obteve RMSE $9.21, R²=0.796 — metricas equivalentes ao melhor run, com cobertura temporal correta.
- Conclusao: o melhor resultado absoluto (RMSE $9.13, MAE $4.72) permanece com 1.6M de Marco (provavelmente por menor distribution shift para Junho, logo apos o corte de treino). Estratificado e mais correto metodologicamente mas nao supera o resultado anterior neste hardware.

### D3 - Decisao: NN com log-transform do target e BatchNorm

- `USE_LOG_TARGET = True`: `log(passenger_fare)` tem distribuicao mais uniforme, reduz o impacto de viagens longas na loss e estabiliza gradientes.
- `BatchNorm1d` apos cada camada oculta: normaliza ativacoes, acelera convergencia e permite learning rate mais alto.
- `ReduceLROnPlateau(patience=3, factor=0.5)`: reduz LR automaticamente ao detectar platô na val_loss. Sem isso o modelo oscilava sem convergir apos epoch 10.
- Arquitetura final: 256 → BN → ReLU → Dropout(0.2) → 128 → BN → ReLU → Dropout(0.2) → 64 → ReLU → 1.
- Progresso de MAE ao aumentar dados de treino:
  - 200k linhas: MAE $6.89
  - 800k linhas: MAE $6.38
  - 1.6M linhas: MAE $4.72 (melhor)

### Selecao do melhor checkpoint da NN

- O melhor resultado (RMSE $9.13, MAE $4.72, R²=0.796) esta em `results/model_comparison.csv` como run_id `a14e4838`.
- O checkpoint salvo em `models/neural_network.pt` corresponde a ultima run executada (estratificada 1.3M, RMSE $9.21).
- Para a apresentacao, as metricas reportadas sao do melhor run historico (a14e4838).
