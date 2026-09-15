"""Airflow DAG: Diabetes ML pipeline (Data Engineering -> Model Engineering -> Deployment).

Runs every 5 minutes. Stage 3 rebuilds Docker images and (re)starts
the FastAPI API and Streamlit app via Docker Compose.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = Path(__file__).resolve().parents[3]
# Interpreter with pipeline dependencies (set PIPELINE_PYTHON if Airflow runs in its own venv)
PYTHON = os.environ.get("PIPELINE_PYTHON", sys.executable)

default_args = {
    "owner": "pmldl",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="diabetes_ml_pipeline",
    description="Data Engineering -> Model Engineering -> Deployment",
    schedule="*/5 * * * *",                 # every 5 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,                      # don't start a new run on top of a long one
    default_args=default_args,
    tags=["pmldl", "mlops", "diabetes"],
) as dag:

    data_engineering = BashOperator(
        task_id="01_data_engineering",
        bash_command=f'cd "{PROJECT_DIR}" && "{PYTHON}" code/datasets/process_data.py',
    )

    model_engineering = BashOperator(
        task_id="02_model_engineering",
        bash_command=f'cd "{PROJECT_DIR}" && "{PYTHON}" code/models/train_model.py',
    )

    deployment = BashOperator(
        task_id="03_deployment",
        bash_command=(
            f'cd "{PROJECT_DIR}" && '
            "docker compose -f code/deployment/docker-compose.yml up -d --build"
        ),
    )

    data_engineering >> model_engineering >> deployment