# Data Dictionary

## Fonte

**Dataset:** NYC Rideshare Raw Data (Kaggle — aaronweymouth)  
**URL:** https://www.kaggle.com/datasets/aaronweymouth/nyc-rideshare-raw-data  
**Origem primária:** NYC TLC HVFHV Trip Records (preprocessados)  
**Arquivo:** `rideshare_data.parquet` (~12 GB, ~365M linhas, Dez/2021–Ago/2023)  
**Operadoras cobertas:** Uber e Lyft (sem Juno/Via)

---

## Camada raw — `rideshare_data.parquet`

Schema real do arquivo conforme inspeção (`spark.read.parquet(...).printSchema()`):

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `business` | string | Operadora: `'Uber'` ou `'Lyft'` |
| `pickup_location` | int | ID de zona TLC de embarque (equivalente a `PULocationID`) |
| `dropoff_location` | int | ID de zona TLC de desembarque |
| `trip_length` | double | Distância da corrida em milhas |
| `total_ride_time` | int | Duração total da corrida em segundos |
| `request_to_pickup` | int | Tempo de espera entre pedido e embarque em segundos |
| `hour_of_day` | int | Hora do embarque (0–23) |
| `month_of_year` | int | Mês do embarque (1–12) |
| `week_of_year` | int | Semana do ano (1–52) |
| `time_of_day` | string | Faixa: `'morning'` / `'afternoon'` / `'evening'` / `'night'` |
| `date` | date | Data do embarque |
| `passenger_fare` | double | **TARGET** — tarifa base do passageiro (USD) |
| `driver_total_pay` | double | Pagamento ao motorista (regulatório TLC) — **não usar como feature** |
| `rideshare_profit` | double | Margem estimada da plataforma — **derivada do target, leakage** |
| `hourly_rate` | double | `passenger_fare / horas` — **leakage direto do target** |
| `dollars_per_mile` | double | `passenger_fare / milhas` — **leakage direto do target** |

### Colunas excluídas do pipeline de ML

| Coluna | Motivo da exclusão |
|--------|--------------------|
| `driver_total_pay` | Determinado por fórmula regulatória TLC (Local Law 150/2018), não por mercado |
| `rideshare_profit` | Derivada de `passenger_fare` — leakage direto |
| `hourly_rate` | Derivada de `passenger_fare` — leakage direto |
| `dollars_per_mile` | Derivada de `passenger_fare` — leakage direto |

---

## Camada bronze — `trips_bronze` (view temporária)

Filtro de escopo temporal aplicado sobre o raw:

```sql
CREATE OR REPLACE TEMPORARY VIEW trips_bronze AS
SELECT * FROM trips_raw
WHERE date >= '2023-03-01'
  AND date <  '2023-09-01'
```

| Propriedade | Valor |
|-------------|-------|
| Período | Março–Agosto 2023 |
| Linhas | ~116.2M |
| Filtro | Apenas recorte de 6 meses necessário para o split temporal |

---

## Camada silver — `trips_silver` (Parquet particionado)

**Path:** `/data/silver/trips_silver/`  
**Particionamento:** `pickup_year_month` (formato `'YYYY-MM'`)  
**Linhas:** ~110.1M (5.26% de drop do bronze pelo DQ gate)

### Limpeza aplicada

| Filtro | Justificativa |
|--------|---------------|
| `passenger_fare > 0` | Tarifa nula indica corrida cancelada ou erro de registro |
| `passenger_fare BETWEEN p0.1% AND p99.9%` | Distribuição heavy-tailed; percentis calibrados via PERCENTILE_APPROX |
| `trip_length BETWEEN p0.1% AND p99.9%` | Idem |
| `total_ride_time BETWEEN p0.1% AND p99.9%` | Idem |
| `speed_mph BETWEEN p0.1% AND p99.9%` | Idem; piso adicional `speed_mph > 0.5` para eliminar GPS travado |
| `request_to_pickup >= 0` | Espera negativa = erro de timestamp |
| `pickup_location NOT IN (264, 265)` | Zonas TLC "Unknown" e "Outside NYC" |
| `dropoff_location NOT IN (264, 265)` | Idem |

Os limites calculados são persistidos em `results/cleaning_bounds.json` para auditoria e reprodutibilidade.

### Schema completo da silver

| Coluna | Tipo | Origem |
|--------|------|--------|
| `date` | date | raw |
| `business` | string | raw |
| `trip_length` | double | raw |
| `total_ride_time` | int | raw |
| `request_to_pickup` | int | raw |
| `hour_of_day` | int | raw |
| `month_of_year` | int | raw |
| `week_of_year` | int | raw |
| `time_of_day` | string | raw |
| `passenger_fare` | double | raw — **TARGET** |
| `speed_mph` | double | calculado: `trip_length / (total_ride_time / 3600.0)` |
| `pickup_dow` | int | calculado: `DAYOFWEEK(date)` (1=Dom, 7=Sáb) |
| `is_weekend` | int | calculado: `pickup_dow IN (1, 7)` |
| `is_rush_hour` | int | calculado: hora 7–9 ou 16–19 |
| `is_late_night` | int | calculado: hora >= 22 ou <= 4 |
| `pickup_airport` | int | calculado: `pickup_location IN (1, 132, 138)` (EWR, JFK, LGA) |
| `dropoff_airport` | int | calculado: `dropoff_location IN (1, 132, 138)` |
| `pu_borough` | string | join com `taxi_zone_lookup.csv` |
| `do_borough` | string | join com `taxi_zone_lookup.csv` |
| `same_borough` | int | calculado: `pu_borough = do_borough` (0 se NULL) |
| `pickup_year_month` | string | calculado: `DATE_FORMAT(date, 'yyyy-MM')` — coluna de partição |

---

## Features usadas nos modelos

Definidas em `notebooks/_lib.py` como `NUMERIC_COLS` e `CATEGORICAL_COLS`.

### Numéricas (14)

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `trip_length` | double | Distância em milhas |
| `total_ride_time` | int | Duração em segundos |
| `request_to_pickup` | int | Tempo de espera em segundos |
| `speed_mph` | double | Velocidade média (calculada) |
| `hour_of_day` | int | Hora do embarque |
| `pickup_dow` | int | Dia da semana (1=Dom, 7=Sáb) |
| `month_of_year` | int | Mês do embarque |
| `week_of_year` | int | Semana do ano |
| `is_weekend` | int | Booleano: final de semana |
| `is_rush_hour` | int | Booleano: horário de pico |
| `is_late_night` | int | Booleano: madrugada |
| `pickup_airport` | int | Booleano: embarque em aeroporto (EWR/JFK/LGA) |
| `dropoff_airport` | int | Booleano: desembarque em aeroporto |
| `same_borough` | int | Booleano: embarque e desembarque no mesmo borough |

### Categóricas (4)

| Feature | Cardinalidade | Valores |
|---------|--------------|---------|
| `business` | 2 | `Uber`, `Lyft` |
| `pu_borough` | 6 | Manhattan, Brooklyn, Queens, Bronx, Staten Island, EWR |
| `do_borough` | 6 | idem |
| `time_of_day` | 4 | `morning`, `afternoon`, `evening`, `night` |

Codificação no pipeline:
- **MLlib (LR, DT):** `StringIndexer` + `VectorIndexer(maxCategories=8)` — VectorIndexer marca os índices como categóricos para que o DecisionTree faça splits não-ordinais.
- **PyTorch (NN):** `pd.get_dummies` com `dummy_na=True` antes de converter para tensores.

---

## Split temporal

| Conjunto | Período | Linhas |
|----------|---------|--------|
| Treino | Mar–Mai 2023 | 56.3M |
| Teste | Jun–Ago 2023 | 53.8M |

`SPLIT_DATE = "2023-06-01"`. Split aleatório (`randomSplit`) é **proibido** — vazaria informação futura no treino.

---

## Lookup de zonas

**Arquivo:** `data/taxi_zone_lookup.csv`  
**Colunas usadas:** `LocationID`, `Borough`, `Zone`  
**Join:** `pickup_location = LocationID` e `dropoff_location = LocationID`  
**Zonas excluídas:** 264 ("Unknown") e 265 ("Outside NYC")

---

## Limitações conhecidas

| Limitação | Impacto |
|-----------|---------|
| Sem latitude/longitude — apenas IDs de zona | Análise geoespacial limitada a nível de borough |
| `time_of_day` é string pré-computado, sem timestamps exatos | Impossível derivar features de hora a partir do raw |
| Apenas Uber e Lyft | Não representa a totalidade do mercado HVFHV de NYC |
| Sem IDs de motorista ou passageiro | Sem análise de frequência por usuário |
| `request_to_pickup` pode ter valores negativos (erro de registro) — filtrados no DQ gate | Reduz levemente a cobertura |
| Viés de sobrevivência: somente corridas concluídas | Cancelamentos não modelados |
