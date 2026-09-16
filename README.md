# PMLDL Assignment 1 — Automated ML Pipeline (Diabetes Prediction)

Automated MLOps pipeline with three stages, orchestrated by **Apache Airflow** and running **every 5 minutes**:

1. **Data Engineering** — load the Pima Indians Diabetes dataset (committed to the repo),
   impute hidden missing values (zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI),
   remove IQR outliers, split 80/20 stratified.
2. **Model Engineering** — interaction feature engineering (PolynomialFeatures), RandomForest training with
   GridSearchCV hyperparameter tuning (5-fold CV, roc_auc), test-set evaluation, metrics and best params
   logged to **MLflow** (SQLite backend), model packaged with joblib.
3. **Deployment** — Docker Compose builds and runs **two separate containers**: a FastAPI model API (:8000)
   and a Streamlit web app (:8501) that calls the API and shows the diabetes-risk prediction.

## Architecture

```
Airflow DAG (every 5 min)
  -> 01_data_engineering   python code/datasets/process_data.py
  -> 02_model_engineering  python code/models/train_model.py   (MLflow + model.pkl)
  -> 03_deployment         docker compose -f code/deployment/docker-compose.yml up -d --build
                             ├── diabetes-api  (FastAPI,   port 8000)
                             └── diabetes-app  (Streamlit, port 8501) -> calls http://api:8000
```

## Repository structure

```
├── code/
│   ├── datasets/process_data.py      # Stage 1
│   ├── models/train_model.py         # Stage 2
│   └── deployment/
│       ├── api/                      # FastAPI + Dockerfile
│       ├── app/                      # Streamlit + Dockerfile
│       └── docker-compose.yml        # Stage 3
├── data/raw/                         # pima-indians-diabetes.csv (auto-downloaded from Kaggle if missing)
├── data/processed/                   # train.csv / test.csv (generated)
├── models/                           # model.pkl / metrics.json (generated)
├── notebooks/                        # EDA
├── services/airflow/dags/            # pipeline DAG
├── start_airflow.sh                  # convenience script to launch Airflow
└── requirements.txt
```

## Prerequisites

- Ubuntu (WSL2) with Python 3.12
- Docker Desktop with WSL integration enabled + Docker Compose v2

## Setup

```bash
git clone https://github.com/<username>/PMLDL_Assignment1.git
cd PMLDL_Assignment1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run stages manually

```bash
python code/datasets/process_data.py                              # Stage 1
python code/models/train_model.py                                 # Stage 2
docker compose -f code/deployment/docker-compose.yml up -d --build   # Stage 3
```

## Run the automated pipeline (every 5 minutes)

Install Airflow once (separate venv to avoid dependency conflicts):

```bash
python3 -m venv airflow-venv
source airflow-venv/bin/activate
export AIRFLOW_HOME="$(pwd)/services/airflow"
export PIPELINE_PYTHON="$(pwd)/venv/bin/python"
pip install "apache-airflow==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
```

Start Airflow (every session):

```bash
source airflow-venv/bin/activate
export AIRFLOW_HOME="$(pwd)/services/airflow"
export PIPELINE_PYTHON="$(pwd)/venv/bin/python"
airflow standalone
```

Open http://localhost:8080 (login: `admin`, password in `services/airflow/standalone_admin_password.txt`),
unpause `diabetes_ml_pipeline` and optionally trigger it manually (▶). Each run refreshes the data,
retrains the model, logs metrics to MLflow and redeploys both containers with the new model.

## Where to see the results of each run

| What                          | Where                                                                    |
|-------------------------------|--------------------------------------------------------------------------|
| Task logs (per run)           | Airflow UI → Grid → click a task square → **Logs**                       |
| Metrics / params / artifacts  | MLflow UI → experiment `diabetes-prediction` (one run per pipeline run)  |
| Latest metrics file           | `models/metrics.json`                                                    |
| Containers rebuilt            | `docker ps` (STATUS uptime resets on every deployment task)              |

## Access

| Service            | URL / command                                                   |
|--------------------|-----------------------------------------------------------------|
| Web app            | http://localhost:8501                                           |
| API (Swagger docs) | http://localhost:8000/docs                                      |
| API health         | http://localhost:8000/health                                    |
| MLflow UI          | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |
| Airflow UI         | http://localhost:8080                                           |

## Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":150,"BloodPressure":80,"SkinThickness":30,"Insulin":100,"BMI":35,"DiabetesPedigreeFunction":0.8,"Age":45}'
```

## Stop everything

```bash
docker compose -f code/deployment/docker-compose.yml down
# Ctrl+C to stop Airflow
```

## Notes

- The pipeline is deterministic (fixed seeds): identical input data produces identical metrics
  across runs, which demonstrates reproducibility of the whole chain.
- `data/processed/*.csv`, `models/model.pkl`, `models/metrics.json` are regenerated by every
  pipeline run; the small model and processed data are committed for convenience.
- `mlflow.db` and `mlartifacts/` (MLflow tracking data) are local-only and not committed.
