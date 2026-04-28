# NYC Rideshare Fare Prediction

Projeto final de Big Data para regressao de tarifa de corridas HVFHV da TLC de Nova York usando PySpark, Spark SQL, MLlib e PyTorch.

## Objetivo

Prever `base_passenger_fare` a partir de sinais temporais, espaciais e operacionais, mantendo um pipeline end-to-end reproduzivel:

- ambiente distribuido com Spark em Docker
- EDA e preprocessing em Spark SQL
- tres modelos para o mesmo target: Linear Regression, Decision Tree e Neural Network

## Stack

- Python 3.10+
- PySpark 3.5
- Docker Compose
- Spark SQL para EDA e ETL
- Spark MLlib para Linear Regression e Decision Tree
- PyTorch para a rede neural

## Estrutura

```text
.
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── Dockerfile.jupyter
├── data/
├── docs/
│   ├── PLAN.md
│   ├── data_dictionary.md
│   └── decisions_log.md
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   └── 05_neural_network.ipynb
├── results/
│   └── model_comparison.csv
└── slides/
```

## Como subir o ambiente

1. Coloque os arquivos Parquet HVFHV em `./data/`.
2. Coloque `taxi_zone_lookup.csv` em `./data/`.
3. Suba o ambiente:

```bash
docker compose up -d --build
docker compose ps
```

4. Valide os servicos:

- Spark UI: `http://localhost:8080`
- Jupyter Lab: `http://localhost:8888`

## Dados esperados em `data/`

- `fhvhv_tripdata_2022-01.parquet` ate `fhvhv_tripdata_2023-08.parquet`
- `taxi_zone_lookup.csv`

Para iteracao inicial, o plano recomenda comecar por `fhvhv_tripdata_2023-08.parquet` e expandir para o range completo depois que o pipeline estiver estavel.

Importante: esse mes isolado serve para smoke test de ingestao e EDA inicial. Os notebooks de modelagem exigem dados antes e depois de `2023-06-01`, senao o split temporal fica sem conjunto de treino ou sem conjunto de teste.

## Ordem recomendada de execucao

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_linear_regression.ipynb`
4. `notebooks/04_decision_tree.ipynb`
5. `notebooks/05_neural_network.ipynb`

## Regras do projeto

- `base_passenger_fare` e o target oficial.
- O split e temporal com `SPLIT_DATE = "2023-06-01"`.
- EDA e ETL devem usar `spark.sql()`, nao DataFrame API.
- Nao usar HDFS, MinIO ou Hadoop.
- Nao trocar `DecisionTreeRegressor` por modelos de ensemble.

## Saidas previstas

- modelos salvos em `models/`
- comparativo de metricas em `results/model_comparison.csv`
- analises finais em `results/`
- decisoes documentadas em `docs/decisions_log.md`
