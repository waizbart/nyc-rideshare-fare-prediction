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
