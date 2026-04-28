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
- Expansao para todo o range 2022-2023 depois do smoke test do pipeline
- Motivo: reduzir custo de iteracao no inicio e diminuir risco de OOM

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
