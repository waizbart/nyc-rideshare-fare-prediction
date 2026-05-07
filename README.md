# NYC Rideshare Fare Prediction

Projeto final da disciplina de Big Data. Pipeline de regressão end-to-end para prever a tarifa de corridas de aplicativo em Nova York, usando PySpark, Spark SQL, MLlib e PyTorch em um cluster Spark containerizado.

Dataset: https://www.kaggle.com/datasets/aaronweymouth/nyc-rideshare-raw-data

---

## Resultados finais

| Modelo | R² | RMSE | MAE | Tempo treino |
|--------|----|------|-----|--------------|
| Linear Regression (MLlib) | 0.806 | $9.07 | $5.76 | 241s |
| Decision Tree depth=8 (MLlib) | **0.813** | **$8.91** | $5.13 | 462s |
| Neural Network (PyTorch) | 0.796 | $9.13 | **$4.72** | 821s |

LR e DT treinam no conjunto completo de 56M corridas (Mar–Mai 2023). A NN usa amostra de 1.6M linhas — limitação de RAM, já que PyTorch não é distribuído.

---

## Dataset

O arquivo `rideshare_data.parquet` (~12 GB, ~365M linhas) é o dataset Kaggle "NYC Rideshare Raw Data" (aaronweymouth), um preprocessamento do HVFHV TLC cobrindo Dez/2021 a Ago/2023. Contém somente Uber e Lyft.

**Recorte usado no projeto:** março a agosto de 2023 (6 meses, ~116M linhas após filtragem bronze).

O split temporal `SPLIT_DATE = "2023-06-01"` divide:
- **Treino:** março, abril, maio 2023 — 56.3M linhas
- **Teste:** junho, julho, agosto 2023 — 53.8M linhas

---

## Pré-requisitos

- Docker + Docker Compose
- ~15 GB de espaço em disco (imagens + dados + silver)
- `rideshare_data.parquet` em `./data/`
- `taxi_zone_lookup.csv` em `./data/` (disponível no repositório da TLC)
- Máquina com no mínimo 8 GB de RAM (recomendado: 16 GB)

---

## Estrutura

```text
.
├── CLAUDE.md                   # contexto persistente do projeto
├── README.md
├── docker-compose.yml          # spark-master + 2 workers + jupyter
├── Dockerfile.jupyter          # imagem com PySpark 3.5.8 + PyTorch
├── data/
│   ├── rideshare_data.parquet  # dataset Kaggle (gitignore)
│   ├── taxi_zone_lookup.csv
│   └── silver/
│       └── trips_silver/       # gerado pelo notebook 02
├── docs/
│   ├── data_dictionary.md
│   ├── decisions_log.md
│   └── PLAN.md
├── models/                     # gerado pelos notebooks (gitignore)
│   ├── linear_regression/
│   ├── decision_tree/
│   └── neural_network.pt
├── notebooks/
│   ├── _lib.py                 # constantes e factory do SparkSession
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   └── 05_neural_network.ipynb
├── results/
│   ├── model_comparison.csv
│   ├── cleaning_bounds.json
│   ├── feature_importance_decision_tree.png
│   ├── residual_analysis_dt.png
│   └── residual_analysis_neural_network.png
└── slides/
    └── apresentacao_final.pdf
```

---

## Subindo o ambiente

```bash
# 1. Coloque rideshare_data.parquet e taxi_zone_lookup.csv em ./data/
# 2. Suba o cluster
docker compose up -d --build

# 3. Verifique os serviços
docker compose ps
# Esperado: 4 containers Up (spark-master, spark-worker-1, spark-worker-2, jupyter)

# 4. Valide o Spark UI (deve listar 2 workers ALIVE com 4G e 2 cores cada)
# http://localhost:8080

# 5. Jupyter Lab (sem token)
# http://localhost:8888
```

---

## Executando os notebooks

Execute em ordem. Cada notebook pode levar de 10 minutos a 2 horas dependendo do hardware.

```bash
# EDA
docker compose exec jupyter jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/01_eda.ipynb

# Preprocessing (silver)
docker compose exec jupyter jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/02_preprocessing.ipynb

# Linear Regression
docker compose exec jupyter jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 \
  notebooks/03_linear_regression.ipynb

# Decision Tree
docker compose exec jupyter jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/04_decision_tree.ipynb

# Neural Network
docker compose exec jupyter jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/05_neural_network.ipynb
```

Tempos estimados (CPU, workers 4G/2 cores cada):

| Notebook | Tempo estimado |
|----------|---------------|
| 01_eda | 10–20 min |
| 02_preprocessing | 30–60 min |
| 03_linear_regression | 5–10 min |
| 04_decision_tree | 8–15 min |
| 05_neural_network | 15–25 min (amostra 1.6M) |

---

## Arquitetura

```
docker-compose:
  spark-master       (Spark UI: localhost:8080)
  spark-worker-1     (4G RAM, 2 cores)
  spark-worker-2     (4G RAM, 2 cores)
  jupyter            (Jupyter Lab: localhost:8888)

Volume ./data montado em /data em todos os containers.
```

O Jupyter container usa a mesma imagem base dos workers (`apache/spark:3.5.8-python3`) como referência de versão, com PyTorch e dependências adicionais instaladas via `Dockerfile.jupyter`. A variável `SPARK_HOME` aponta para o PySpark 3.5.8 instalado via pip, garantindo compatibilidade de serialização com os workers.

---

## Convenções

- **ETL/EDA:** Spark SQL puro via `spark.sql()` — sem DataFrame API para data prep.
- **ML:** Spark MLlib (LR, DT) e PyTorch (NN).
- **Split:** temporal com `SPLIT_DATE = "2023-06-01"` — sem `randomSplit`.
- **Seed:** `SEED = 42` em toda aleatoriedade.
- **Constantes compartilhadas:** `notebooks/_lib.py` — fonte única de verdade para features, paths e configuração do SparkSession.

---

## Saídas geradas

| Arquivo | Descrição |
|---------|-----------|
| `data/silver/trips_silver/` | Silver layer particionada por `pickup_year_month` |
| `results/cleaning_bounds.json` | Percentis 0.1%/99.9% usados no DQ gate |
| `results/model_comparison.csv` | Todas as runs com run_id, timestamp, métricas |
| `results/feature_importance_decision_tree.png` | Top 15 features da árvore |
| `results/residual_analysis_dt.png` | Resíduos do Decision Tree |
| `results/residual_analysis_neural_network.png` | Resíduos e curva de loss da NN |
| `models/linear_regression/` | Pipeline MLlib serializado |
| `models/decision_tree/` | Pipeline MLlib serializado |
| `models/neural_network.pt` | Checkpoint PyTorch (weights + scaler + feature columns) |
