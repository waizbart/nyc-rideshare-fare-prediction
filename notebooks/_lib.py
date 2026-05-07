"""Utilitarios compartilhados entre os notebooks do projeto.

Centraliza constantes, factory de SparkSession e o append em
model_comparison.csv. Importar nos notebooks evita duplicacao e drift entre
modelos (alterar uma feature em um notebook sem propagar para os outros).

Uso tipico no inicio de um notebook:

    import sys
    sys.path.insert(0, '/home/jovyan/work/notebooks')
    from _lib import build_spark, SEED, SPLIT_DATE, SILVER_PATH, ...

    spark = build_spark('nome-do-app')

Schema real do arquivo rideshare_data.parquet (Kaggle NYC Rideshare Raw Data,
preprocessado a partir do HVFHV TLC). Colunas relevantes:
  business           - 'Uber' ou 'Lyft'
  pickup_location    - ID de zona TLC (== PULocationID no HVFHV bruto)
  dropoff_location   - ID de zona TLC
  trip_length        - distancia em milhas
  total_ride_time    - duracao da corrida em segundos
  request_to_pickup  - tempo de espera em segundos
  hour_of_day        - hora do embarque (0-23)
  month_of_year      - mes do embarque (1-12)
  week_of_year       - semana do ano
  time_of_day        - 'morning'/'afternoon'/'evening'/'night'
  date               - data do embarque (tipo date)
  passenger_fare     - TARGET: tarifa base do passageiro (USD)
  driver_total_pay   - paga ao motorista (regulatorio TLC, nao usar como feature)
  rideshare_profit   - margem estimada da plataforma (nao usar: derivada do target)
  hourly_rate        - passenger_fare / horas (LEAKAGE: derivada do target)
  dollars_per_mile   - passenger_fare / milhas (LEAKAGE: derivada do target)
"""
from __future__ import annotations

import datetime as _dt
import subprocess as _sp
import uuid as _uuid
from pathlib import Path

SEED = 42
SPLIT_DATE = '2023-06-01'

DATA_GLOB = '/data/rideshare_data.parquet'
LOOKUP_PATH = '/data/taxi_zone_lookup.csv'
SILVER_PATH = '/data/silver/trips_silver'

RESULTS_DIR = Path('/results')
RESULTS_PATH = RESULTS_DIR / 'model_comparison.csv'
BOUNDS_PATH = RESULTS_DIR / 'cleaning_bounds.json'

TARGET_COL = 'passenger_fare'

# Colunas numericas seguras (sem leakage do target).
# Ausentes do dataset original e calculados no preprocessing:
#   speed_mph   = trip_length / (total_ride_time / 3600)
#   pickup_dow  = DAYOFWEEK(date)
#   is_weekend / is_rush_hour / is_late_night = booleanos de tempo
#   pickup_airport / dropoff_airport / same_borough = de zona lookup
NUMERIC_COLS = [
    'trip_length',       # milhas
    'total_ride_time',   # segundos
    'request_to_pickup', # espera em segundos
    'speed_mph',         # calculado no preprocessing
    'hour_of_day',       # ja no dataset
    'pickup_dow',        # calculado de date
    'month_of_year',     # ja no dataset
    'week_of_year',      # ja no dataset
    'is_weekend',        # calculado
    'is_rush_hour',      # calculado
    'is_late_night',     # calculado
    'pickup_airport',    # calculado (zonas 1, 132, 138)
    'dropoff_airport',   # calculado
    'same_borough',      # calculado via lookup
]
CATEGORICAL_COLS = ['business', 'pu_borough', 'do_borough', 'time_of_day']


def build_spark(app_name: str, executor_memory: str = '3g', driver_memory: str = '4g'):
    """SparkSession com configuracao consistente entre notebooks.

    shuffle.partitions=32 adequado para ~100M linhas / 6 meses.
    AQE habilitado explicitamente para documentar intencao.
    executor_memory/driver_memory podem ser sobrescritos pelo notebook 05 para
    liberar mais heap no driver antes do toPandas() de grande volume.
    """
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
        .appName(app_name)
        .master('spark://spark-master:7077')
        .config('spark.executor.memory', executor_memory)
        .config('spark.driver.memory', driver_memory)
        .config('spark.sql.shuffle.partitions', '32')
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
        .config('spark.network.timeout', '600s')
        .config('spark.executor.heartbeatInterval', '60s')
        .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    return spark


def _git_sha() -> str:
    """Retorna SHA curto do HEAD; tolera ausencia do binario `git` na imagem."""
    try:
        return _sp.check_output(
            ['git', '-C', '/home/jovyan/work', 'rev-parse', '--short', 'HEAD'],
            stderr=_sp.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def append_metric(model: str, rmse: float, mae: float, r2: float,
                  train_seconds: float, notes: str = ''):
    """Append-only em model_comparison.csv, preserva historico de runs.

    Inclui run_id, timestamp_utc e git_sha em cada linha.
    """
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        'run_id': _uuid.uuid4().hex[:8],
        'timestamp_utc': _dt.datetime.utcnow().isoformat(timespec='seconds'),
        'git_sha': _git_sha(),
        'model': model,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_seconds': train_seconds,
        'notes': notes,
    }
    columns = ['run_id', 'timestamp_utc', 'git_sha', 'model',
               'rmse', 'mae', 'r2', 'train_seconds', 'notes']
    df = pd.DataFrame([row], columns=columns)
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        existing = pd.read_csv(RESULTS_PATH)
        for col in columns:
            if col not in existing.columns:
                existing[col] = pd.NA
        df = pd.concat([existing[columns], df], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)
    return df
