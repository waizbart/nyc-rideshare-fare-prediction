# Plano de Desenvolvimento, Projeto Final Big Data
## NYC Rideshare Regression com PySpark

---

## 1. Contexto e premissas

**Stack travada**
- Python + PySpark 3.5
- Docker Compose com Spark master + 2 workers + Jupyter
- Sem HDFS, sem MinIO, sem Hadoop. Volume local montado nos containers.
- Manipulação de dados via Spark SQL (`spark.sql()`), não via DataFrame API.
- ML via Spark MLlib (Decision Tree, Linear Regression) e PyTorch para a Rede Neural.

**Dataset**
- NYC TLC HVFHV Trip Records (2022-2023), formato Parquet.
- Tamanho real: cerca de 12-15 GB compactado, ~360M linhas.
- Taxi Zone Lookup CSV para enriquecimento de borough.

**Tarefa**
- Regressão sobre `base_passenger_fare`.
- Justificativa documentada: `driver_pay` é regulatorialmente determinado pela Local Law 150/2018, gerando regressão trivial.

**Entregáveis obrigatórios pelo enunciado**
- Parte 1: Ambiente Big Data (1 ponto)
- Parte 2: EDA + preprocessing (5 pontos, peso maior)
- Parte 3: 3 modelos preditivos (4 pontos)
- Apresentação final

---

## 2. Timeline, 4 semanas

Assumindo apresentação em 27/05. Se for em 11/05 ou 13/05, comprimir Semanas 3 e 4 em uma só, e usar amostra menor (1 mês ao invés de full).

### Semana 1, 28/04 a 04/05: Infra + ingestão + EDA inicial

| Dia | Atividade | Responsável sugerido |
|---|---|---|
| 28-29/04 | Subir Docker, validar cluster Spark, baixar Parquet | Pessoa A |
| 30/04 | Schema validation, contagem por mês, range temporal | Pessoa B |
| 01-02/05 | EDA univariada: nulos, distribuições, outliers visíveis | Pessoa B |
| 03-04/05 | EDA bivariada: tarifa vs. distância, tempo, hora, zona | Pessoa C |

**Marco S1**: Notebook EDA fechado com decisões de cleaning justificadas.

### Semana 2, 05/05 a 11/05: Preprocessing + 2 modelos

| Dia | Atividade | Responsável |
|---|---|---|
| 05-06/05 | Camada silver via SQL: cleaning + feature engineering | Pessoa B + C |
| 07/05 | Split temporal, validação do split | Pessoa C |
| 08-09/05 | Linear Regression baseline + métricas | Pessoa C |
| 10-11/05 | Decision Tree + análise de feature importance | Pessoa D |

**Marco S2**: Tabela comparativa LR vs DT em `base_passenger_fare`.

### Semana 3, 12/05 a 18/05: Rede Neural + tuning

| Dia | Atividade | Responsável |
|---|---|---|
| 12-13/05 | Setup PyTorch, dataloader a partir de Parquet | Pessoa D |
| 14-15/05 | Modelo NN: arquitetura simples (3 hidden layers, ReLU) | Pessoa D |
| 16/05 | Treino com early stopping em validation set | Pessoa D |
| 17-18/05 | Tuning leve dos 3 modelos, escolher hiperparâmetros finais | Time todo |

**Marco S3**: 3 modelos treinados, métricas calculadas no test set.

### Semana 4, 19/05 a 27/05: Análise final + slides + demo

| Dia | Atividade | Responsável |
|---|---|---|
| 19-20/05 | Análise de resíduos, erro estratificado por borough/distância | Pessoa B |
| 21/05 | Relatório de conclusões | Pessoa A + B |
| 22-23/05 | Slides | Pessoa A + D |
| 24-25/05 | Ensaio da apresentação, gravação de demo | Time todo |
| 26-27/05 | Apresentação final | Time todo |

---

## 3. Detalhamento por parte

### Parte 1, Ambiente Big Data, 1 ponto

**Status atual**: docker-compose.yml pronto.

**Itens pendentes**
1. Subir o stack e confirmar 2 workers ALIVE no Spark UI (`localhost:8080`).
2. Baixar arquivos Parquet do TLC para `./data/`. Lista mensal: `fhvhv_tripdata_2022-01.parquet` até `fhvhv_tripdata_2023-08.parquet`.
3. Baixar `taxi_zone_lookup.csv` do TLC.
4. Estender a imagem do Jupyter para incluir PyTorch (Dockerfile customizado), ou usar `!pip install torch torchvision` no notebook na primeira célula.
5. Documento README explicando como subir o ambiente, incluindo prints do Spark UI funcionando.

**Definition of Done**
- `docker compose up -d` sobe limpo
- Spark UI mostra master + 2 workers
- Jupyter acessível em `localhost:8888`
- Query SQL de teste retorna 4 operadoras (HV0002 a HV0005)

### Parte 2, Análise de dados, 5 pontos (peso maior, atenção)

**Sub-entregáveis**

#### 2.1 EDA, notebook `01_eda.ipynb`
Tudo em Spark SQL. Pandas só para os plots, depois de `.toPandas()` em agregados pequenos.

Tópicos obrigatórios:
- Volumetria total e por operadora
- Cobertura temporal (dia mais antigo, dia mais recente, dias faltantes)
- Nulos por coluna (especial atenção a `on_scene_datetime`, costuma ter cobertura desigual)
- Distribuição univariada de `trip_miles`, `trip_time`, `base_passenger_fare`, `tips`, `driver_pay`
- Outliers identificados (velocidade impossível, duração negativa, etc.)
- Volume por hora do dia, dia da semana, mês
- Top 10 zonas de pickup e dropoff
- Matriz origem-destino entre boroughs
- Correlação entre features numéricas
- Sazonalidade

#### 2.2 Preprocessing, notebook `02_preprocessing.ipynb`
SQL puro com CTEs e `CREATE OR REPLACE TEMPORARY VIEW`. Sem `df.filter()`.

Etapas:
1. Carregamento dos Parquet em view `trips_bronze`
2. Validação temporal (pickup >= request, dropoff > pickup)
3. Validação física (distância > 0, tempo > 60s, velocidade plausível)
4. Validação financeira (tarifa > 0, percentis 0,1% e 99,9%)
5. Filtro de zonas inválidas (264, 265)
6. Feature engineering em SQL: `pickup_hour`, `pickup_dow`, `pickup_month`, `is_weekend`, `is_rush_hour`, `is_late_night`, `is_airport_pickup`, `is_airport_dropoff`, `same_borough`, `wait_time_sec`, flags Y/N -> 0/1
7. Join com `taxi_zone_lookup` para borough
8. Salvar camada silver em Parquet particionado por ano-mês

**Definition of Done**
- EDA cobre 4 dimensões (temporal, espacial, financeira, operacional)
- Cada filtro tem justificativa numérica (ex: "5,2% das linhas removidas por velocidade fora do percentil 0,5-99,5")
- Camada silver materializada em disco

### Parte 3, Modelos preditivos, 4 pontos

#### 3.1 Linear Regression (MLlib)
Notebook `03_linear_regression.ipynb`. Baseline obrigatório, simples e rápido.

#### 3.2 Decision Tree Regressor (MLlib)
Notebook `04_decision_tree.ipynb`. Atenção: o enunciado pede árvore, não boosting. Usar `DecisionTreeRegressor`, não `GBTRegressor` ou `RandomForestRegressor`.

#### 3.3 Rede Neural (PyTorch)
Notebook `05_neural_network.ipynb`.

Arquitetura sugerida:
- Input: features numéricas + one-hot das categóricas
- Hidden 1: 128 neurônios, ReLU, Dropout 0,2
- Hidden 2: 64 neurônios, ReLU, Dropout 0,2
- Hidden 3: 32 neurônios, ReLU
- Output: 1 neurônio (regressão)
- Loss: MSE, Optimizer: Adam, lr 1e-3

Estratégia de dados: amostra de 1-2M linhas para treino. PyTorch puro, não distribuído. É defensável porque o foco da NN não é throughput, é demonstrar a capacidade do modelo.

**Métricas obrigatórias para todos os 3 modelos**
- RMSE (USD)
- MAE (USD)
- R²
- Tempo de treino
- Tabela comparativa final

**Análises adicionais que ganham pontos**
- Erro estratificado por borough
- Erro estratificado por bucket de distância (0-2 mi, 2-5 mi, 5-10 mi, 10+ mi)
- Análise de resíduos (histograma, residuos vs predito)
- Feature importance (DT)
- Coeficientes (LR)

**Definition of Done**
- 3 modelos treinados no mesmo split temporal
- Tabela comparativa em `results/model_comparison.csv`
- Análise de resíduos por modelo

---

## 4. Apresentação

**Duração estimada**: 15-20 minutos

**Estrutura sugerida**
1. Contexto e problema (1 slide)
2. Dataset escolhido + justificativa (1 slide)
3. Arquitetura Big Data (1 slide com diagrama)
4. EDA, principais insights (3-4 slides com gráficos)
5. Preprocessing, decisões e impacto (1-2 slides)
6. Modelos e resultados (3 slides)
7. Demo prática (3-5 minutos ao vivo)
8. Desafios enfrentados (1 slide, sincero)
9. Melhorias futuras (1 slide)
10. Q&A

**Demo prática**
- Notebook pronto, dados pré-carregados
- Subir Spark, mostrar UI com workers
- Query SQL EDA retornando ao vivo
- Predição usando modelo salvo (NN ou DT) com input novo

---

## 5. Estrutura do repositório

```
projeto-bigdata-rideshare/
├── docker-compose.yml
├── Dockerfile.jupyter           # com PyTorch
├── README.md                    # como rodar
├── data/                        # Parquet + zones (gitignore)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   └── 05_neural_network.ipynb
├── models/                      # modelos salvos (gitignore)
├── results/
│   ├── model_comparison.csv
│   ├── residual_analysis.png
│   └── feature_importance.png
├── slides/
│   └── apresentacao_final.pdf
└── docs/
    ├── data_dictionary.md
    └── decisions_log.md         # justificativas de cleaning, escolha de target, etc
```

---

## 6. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Volume de dados estoura RAM dos laptops | Alta | Alto | Trabalhar com 2 meses primeiro, full só na entrega |
| `shared_match_flag` tem semântica diferente entre Lyft e Uber | Confirmado | Médio | Documentar a limitação ou tratar como flag bruta sem comparação direta |
| Cobertura desigual de `on_scene_datetime` | Confirmado | Baixo | Não usar como feature crítica |
| NN não converge ou overfita rápido | Média | Alto | Early stopping + dropout + escala das features (StandardScaler) |
| Demo da apresentação falha ao vivo | Média | Alto | Gravar fallback em vídeo |
| Modelo de DT explode em profundidade | Baixa | Médio | `maxDepth=10` máximo |
| Spark workers caem por OOM | Média | Médio | `spark.sql.shuffle.partitions=200`, sample para iteração |
| Limitação de SQL puro em features categóricas de alta cardinalidade | Média | Baixo | Frequency encoding em SQL, não target encoding |

---

## 7. Decisões pendentes

Antes de começar a Semana 1, fechar:

1. PyTorch via Dockerfile customizado ou `!pip install` no notebook?
2. Equipe é de 3 ou 4 pessoas? Se 3, distribuir Pessoa D entre A, B e C.
3. Apresentação em qual data (11, 13, 25 ou 27/05)? Define se o plano é de 2 ou 4 semanas.
4. Vai usar todo o range 2022-2023 ou recortar (ex: só 2023)? Recorte é válido e mais leve.
5. Versionamento: GitHub público (bom para portfólio) ou privado?

---

## 8. Critérios de qualidade que diferenciam nota máxima

Itens que normalmente faltam em projetos da disciplina e que aumentam nota:

1. **Justificativa numérica em cada filtro de cleaning**, não só "removemos outliers"
2. **Split temporal**, não random. Documentar por quê.
3. **Erro estratificado**, não só RMSE médio. Mostrar onde o modelo erra mais.
4. **Reprodutibilidade**: seeds fixos, versões pinadas no Dockerfile
5. **Análise de resíduos**, não só métricas
6. **Discussão crítica do limite dos modelos**: até onde dá para prever fare sem feature de surge?
7. **Documentação de decisões em `decisions_log.md`** ao longo do projeto, não no fim

---

## 9. Próximas ações imediatas

1. Subir Docker e validar cluster
2. Baixar 1 mês de Parquet (`fhvhv_tripdata_2023-08.parquet`) para começar leve
3. Criar repositório com a estrutura do item 5
4. Decidir os pontos do item 7
5. Começar `01_eda.ipynb` em Spark SQL
