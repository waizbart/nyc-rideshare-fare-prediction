# Data Dictionary

## Fonte

Dataset HVFHV da NYC TLC, com uma linha por viagem em formato Parquet.

- Range bruto disponivel na TLC: janeiro/2022 a agosto/2023.
- Recorte em uso neste projeto: marco/2023 a agosto/2023 (6 meses), suficiente para o split temporal `SPLIT_DATE = "2023-06-01"`.

## Colunas principais

| Coluna | Tipo | Descricao |
|---|---|---|
| `hvfhs_license_num` | string | Operadora: HV0002 Juno, HV0003 Uber, HV0004 Via, HV0005 Lyft |
| `dispatching_base_num` | string | Base que despachou a corrida |
| `originating_base_num` | string | Base que recebeu a solicitacao |
| `request_datetime` | timestamp | Horario do pedido |
| `on_scene_datetime` | timestamp | Horario de chegada do motorista |
| `pickup_datetime` | timestamp | Inicio da corrida |
| `dropoff_datetime` | timestamp | Fim da corrida |
| `PULocationID` | int | Zona TLC de pickup |
| `DOLocationID` | int | Zona TLC de dropoff |
| `trip_miles` | float | Distancia em milhas |
| `trip_time` | int | Duracao em segundos |
| `base_passenger_fare` | float | Target principal do projeto |
| `tolls` | float | Pedagios |
| `bcf` | float | Black Car Fund |
| `sales_tax` | float | Imposto estadual |
| `congestion_surcharge` | float | Taxa de congestionamento |
| `airport_fee` | float | Taxa de aeroporto |
| `tips` | float | Gorjeta |
| `driver_pay` | float | Pagamento ao motorista |
| `shared_request_flag` | string | `Y` se o usuario aceitou corrida compartilhada |
| `shared_match_flag` | string | `Y` se houve match em corrida compartilhada |
| `access_a_ride_flag` | string | Corrida via MTA Access-A-Ride |
| `wav_request_flag` | string | Solicitacao de veiculo acessivel |
| `wav_match_flag` | string | Corrida em veiculo acessivel |

## Features derivadas previstas na silver

- `pickup_hour`
- `pickup_dow`
- `pickup_month_num`
- `pickup_year`
- `pickup_year_month`
- `is_weekend`
- `is_rush_hour`
- `is_late_night`
- `pickup_airport`
- `dropoff_airport`
- `pu_borough`
- `do_borough`
- `same_borough`
- `wait_time_sec`
- `speed_mph`
- `shared_req`
- `wav_req`

## Limitacoes criticas

- `driver_pay` e regulatorio, nao deve virar target.
- `on_scene_datetime` tem cobertura desigual e nao deve ser feature critica.
- `shared_match_flag` nao e comparavel diretamente entre Uber e Lyft.
- `PULocationID` e `DOLocationID` 264 e 265 devem ser removidos das analises geograficas.
- Nao existem identificadores de motorista ou passageiro.

