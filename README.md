# 🎓 Passos Mágicos – Predição de Defasagem Escolar

Projeto de Machine Learning desenvolvido para prever o nível de defasagem escolar de estudantes da **Associação Passos Mágicos**.

O sistema utiliza indicadores educacionais e psicossociais para classificar estudantes em três níveis de risco:

* **Em fase**
* **Moderada**
* **Severa**

A solução inclui:

* Pipeline completo de Machine Learning
* API REST para predições
* Containerização com Docker
* Testes automatizados
* Sistema de monitoramento e logging

---

# 📊 Arquitetura do Projeto

```
Dados PEDE
      ↓
Pré-processamento
      ↓
Feature Engineering
      ↓
Treinamento de múltiplos modelos
      ↓
Seleção do melhor modelo
      ↓
Serialização do modelo
      ↓
API FastAPI
      ↓
Predições em produção
      ↓
Monitoramento e logging
```

---

# 📁 Estrutura do Repositório

```
passos-magicos-ml/

data/
 ├── raw/
 └── processed/

models/
 ├── model.pkl
 ├── scaler.pkl
 ├── label_encoder.pkl
 ├── model_metadata.json
 └── evaluation_report.txt

src/
 ├── preprocessing.py
 ├── features.py
 ├── train.py
 ├── evaluation.py
 ├── utils.py
 └── model_test.py

app/
 ├── main.py
 ├── predictor.py
 └── schemas.py

tests/

monitoring/
 ├── drift_monitor.py
 └── dashboard.py

logs/
 ├── api.log
 ├── system.log
 └── predictions.csv

Dockerfile
requirements.txt
run_pipeline.py
README.md
```

---

# 📊 Dataset

Fonte dos dados:

**Pesquisa Extensiva do Desenvolvimento Educacional (PEDE)**
Associação Passos Mágicos

Os dados incluem:

* Indicadores educacionais
* Indicadores psicossociais
* Engajamento escolar
* Progresso educacional

Período utilizado:

```
2020
2021
2022
```

---

# ⚙️ Pipeline de Machine Learning

## 1️⃣ Pré-processamento

O módulo `preprocessing.py` realiza:

* Consolidação de dados de múltiplos anos
* Tratamento de valores faltantes
* Encoding de variáveis categóricas
* Normalização de variáveis numéricas
* Split treino/teste

Divisão utilizada:

```
80% treino
20% teste
```

---

## 2️⃣ Feature Engineering

O módulo `features.py` cria novas variáveis baseadas em:

* Desempenho acadêmico
* Engajamento do estudante
* Evolução temporal
* Indicadores psicossociais
* Interações entre variáveis

Mais de **20 features derivadas** são criadas.

---

## 3️⃣ Treinamento de Modelos

Foram treinados múltiplos algoritmos:

* Logistic Regression
* Random Forest
* Decision Tree
* Gradient Boosting
* XGBoost
* LightGBM

A seleção do melhor modelo foi feita utilizando:

```
F1 Score Weighted
```

---

## 4️⃣ Modelo Final

Modelo selecionado:

```
Decision Tree
```

Métricas obtidas:

```
Accuracy: 1.00
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
```

---

# 📡 API de Predição

A API foi desenvolvida utilizando **FastAPI**.

## Endpoint principal

```
POST /predict
```

Exemplo de requisição:

```json
{
 "RA": "12345",
 "Ano ingresso": 2020,
 "Fase": "Fase 3",
 "Turma": "Turma A",
 "Gênero": "Feminino",
 "Instituição de ensino": "Escola Municipal",
 "IAA": 7.5,
 "IEG": 8.0,
 "IPS": 6.5,
 "IDA": 7.0,
 "IPV": 8.5,
 "IAN": 7.8
}
```

Resposta:

```json
{
 "prediction": "Em fase",
 "confidence": 0.85,
 "probabilities": {
   "Em fase": 0.85,
   "Moderada": 0.10,
   "Severa": 0.05
 }
}
```

---

# 🐳 Executando com Docker

Build da imagem:

```
docker build -t passos-magicos-api .
```

Executar container:

```
docker run -p 5000:5000 passos-magicos-api
```

API disponível em:

```
http://localhost:5000
```

---

# 📊 Monitoramento do Modelo

O projeto inclui um sistema de **monitoramento do modelo em produção**, permitindo acompanhar o comportamento da API, das predições realizadas e possíveis mudanças na distribuição dos dados.

O monitoramento é composto por três componentes principais:

- **Sistema de logging**
- **Detecção de data drift**
- **Dashboard de monitoramento**

---

# 📝 Sistema de Logging

Os logs são registrados **na camada da API (`main.py`)**, garantindo que cada requisição de predição seja rastreável.

A classe responsável pela inferência (`predictor.py`) **não grava logs diretamente**, mantendo separação clara de responsabilidades.

Estrutura de logs gerados:

logs/
 ├── api.log
 ├── system.log
 └── predictions.csv

---

## 📡 API Logs

Registram eventos relacionados às requisições da API:

- requisições recebidas
- execução do endpoint `/predict`
- erros da aplicação

Arquivo:
      logs/api.log

Exemplo:
```
2026-03-11 13:05:21 INFO Recebendo requisição de predição RA=20001
2026-03-11 13:05:21 INFO Predição realizada: Em fase
```

---

## ⚙️ System Logs

Registram eventos internos do sistema:

- inicialização da API
- carregamento do modelo
- erros internos da aplicação

Arquivo:
      logs/system.log
      
---

## 📈 Prediction Logs

Cada predição realizada pela API é registrada no arquivo:

logs/predictions.csv

Campos registrados:

- timestamp
- RA do estudante
- classe prevista
- confiança da predição

Exemplo:
```
timestamp,RA,prediction,confidence
2026-03-11T13:10:12,12345,Em fase,0.92
2026-03-11T13:10:20,12346,Moderada,0.71
```

Esse log é utilizado para:

- auditoria de predições
- análise de comportamento do modelo
- monitoramento de drift

---

# 📉 Monitoramento de Data Drift

O sistema implementa detecção de **data drift**, comparando a distribuição dos dados utilizados no treinamento com os dados recebidos em produção.

A detecção utiliza o **teste estatístico Kolmogorov-Smirnov (KS Test)**.

Arquivo responsável:
      monitoring/drift_monitor.py

O sistema compara:
dados de treino
vs
dados recebidos em produção

Se a distribuição de alguma variável mudar significativamente, um alerta de drift é gerado.

Critério utilizado:
p-value < 0.05 → drift detectado

---

# 📊 Dashboard de Monitoramento

O projeto inclui um dashboard interativo desenvolvido em **Streamlit** para acompanhar o comportamento do modelo em produção.

Arquivo:
monitoring/dashboard.py

Executar o dashboard:

```bash
streamlit run monitoring/dashboard.py
```

---
## 📊 Informações exibidas no Dashboard

O dashboard apresenta:

### 📈 Métricas principais

- total de predições realizadas
- confiança média do modelo
- última classe prevista

### 📊 Visualizações

- distribuição das classes previstas
- distribuição da confiança das predições
- histórico das últimas predições

### 📝 Logs do sistema

- últimas entradas do `api.log`
- últimas entradas do `system.log`

### 📉 Monitoramento de Drift

- tabela com *p-values* por feature
- gráfico de detecção de drift
- alerta visual caso drift seja detectado

---

## 🎯 Benefícios do Monitoramento

O sistema de monitoramento permite:

- rastrear todas as predições realizadas
- detectar mudanças nos dados em produção
- identificar possíveis problemas no modelo
- acompanhar o comportamento da API

Essas práticas seguem princípios de **MLOps**, garantindo maior confiabilidade e observabilidade do sistema de Machine Learning em produção.
# 🧪 Testes Automatizados

Os testes foram implementados utilizando **pytest**.

Cobertura alcançada:

```
89%
```

O requisito mínimo do projeto era **80%**.

---

# 🚀 Deploy

A aplicação pode ser implantada em:

* AWS EC2
* Google Cloud Run
* Azure App Service
* Heroku

---

# 👥 Equipe

Projeto desenvolvido para o **Datathon FIAP – Passos Mágicos**

Equipe:

* Caio Marinho – Machine Learning Pipeline
* Natasha Bomfim – API & Deployment
* Fernanda Pavão – Testes & Qualidade
* André Almeida – Monitoramento, Documentação e Apresentação

---

# 📄 Licença

Uso educacional – Datathon FIAP.
