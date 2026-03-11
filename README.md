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

Para garantir confiabilidade em produção foram implementados mecanismos de monitoramento:

### Monitoramento de Drift

O sistema monitora:

* Data Drift
* Prediction Drift
* Performance Drift

### Logging

Logs são registrados para:

* requisições da API
* erros
* predições realizadas
* tempo de resposta

---

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
