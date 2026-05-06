"""Utilitarios compartilhados entre os notebooks do projeto.

Centraliza constantes, factory de SparkSession e o append em
model_comparison.csv. Importar nos notebooks evita duplicacao e drift entre
modelos (alterar uma feature em um notebook sem propagar para os outros).

Uso tipico no inicio de um notebook:

    import sys
    sys.path.insert(0, '/home/jovyan/work/notebooks')
    from _lib import build_spark, SEED, SPLIT_DATE, SILVER_PATH, ...

    spark = build_spark('nome-do-app')
"""
from __future__ import annotations

import datetime as _dt
import subprocess as _sp
import uuid as _uuid
from pathlib import Path

SEED = 42
SPLIT_DATE = '2023-06-01'

DATA_GLOB = '/data/fhvhv_tripdata_*.parquet'
LOOKUP_PATH = '/data/taxi_zone_lookup.csv'
SILVER_PATH = '/data/silver/trips_silver'

RESULTS_DIR = Path('/results')
RESULTS_PATH = RESULTS_DIR / 'model_comparison.csv'
BOUNDS_PATH = RESULTS_DIR / 'cleaning_bounds.json'

TARGET_COL = 'base_passenger_fare'

NUMERIC_COLS = [
    'trip_miles', 'trip_time', 'wait_time_sec', 'speed_mph', 'pickup_hour',
    'pickup_dow', 'pickup_month_num', 'is_weekend', 'is_rush_hour',
    'is_late_night', 'pickup_airport', 'dropoff_airport', 'same_borough',
    'shared_req', 'wav_req',
]
CATEGORICAL_COLS = ['hvfhs_license_num', 'pu_borough', 'do_borough']


def build_spark(app_name: str):
    """SparkSession com configuracao consistente entre notebooks.

    shuffle.partitions=32 e adequado ao volume atual (~3 GB / 6 meses).
    AQE explicito porque o default do 3.5 ja habilita, mas documenta intencao
    e protege contra mudanca de default em upgrades.
    """
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
        .appName(app_name)
        .master('spark://spark-master:7077')
        .config('spark.executor.memory', '3g')
        .config('spark.driver.memory', '4g')
        .config('spark.sql.shuffle.partitions', '32')
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
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

    Inclui run_id, timestamp_utc e git_sha em cada linha. Substitui o
    upsert manual anterior, que sobrescrevia execucoes da mesma familia.
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
        # CSV legado pode nao ter as colunas novas; reconcilia antes de concat.
        for col in columns:
            if col not in existing.columns:
                existing[col] = pd.NA
        df = pd.concat([existing[columns], df], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)
    return df
