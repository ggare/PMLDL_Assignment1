import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = PROJECT_DIR / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_DIR / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODELS_DIR / "model.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

MLFLOW_DB = PROJECT_DIR / "mlflow.db"
ARTIFACTS_DIR = PROJECT_DIR / "mlartifacts"
EXPERIMENT_NAME = "diabetes-prediction"

TARGET = "Class"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
RANDOM_STATE = 42

PARAM_GRID = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 8],
    "classifier__min_samples_leaf": [1, 5],
    "classifier__class_weight": [None, "balanced"],
}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("interactions", PolynomialFeatures(degree=2, interaction_only=True,
                                             include_bias=False)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def tune_model(X_train, y_train) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"Best CV roc_auc : {search.best_score_:.4f}")
    print(f"Best params     : {search.best_params_}")
    return search


def setup_mlflow() -> str:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return client.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACTS_DIR.as_uri()
        )
    return experiment.experiment_id


def main() -> None:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    search = tune_model(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print("Test metrics:", metrics)

    # Package the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {MODEL_FILE}, metrics to {METRICS_FILE}")

    # Log everything to MLflow
    experiment_id = setup_mlflow()
    with mlflow.start_run(run_name="random_forest_tuned", experiment_id=experiment_id):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("feature_engineering", "polynomial_interactions")
        mlflow.log_param("tuning", "GridSearchCV_5fold_roc_auc")
        mlflow.log_params({k.replace("classifier__", ""): str(v)
                           for k, v in search.best_params_.items()})
        mlflow.log_metric("cv_roc_auc", round(search.best_score_, 4))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(METRICS_FILE))
    print(f"Metrics and artifacts logged to MLflow (db: {MLFLOW_DB})")


if __name__ == "__main__":
    main()