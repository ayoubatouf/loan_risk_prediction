from src.pipelines.training_pipeline import run_training_pipeline
import os
from config.paths import *
import joblib
import pandas as pd
import json
from src.visualization.metrics import visualize_metrics


def run_training_script(train_path, test_path, model_path, metrics_path, params_path):

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.drop("loan_status", axis=1)
    y_train = train_data["loan_status"]
    X_test = test_data.drop("loan_status", axis=1)
    y_test = test_data["loan_status"]

    with open(params_path, "r") as f:
        best_params = json.load(f)

    model, metrics, y_pred, y_pred_proba = run_training_pipeline(
        X_train, X_test, y_train, y_test, best_params
    )

    joblib.dump(model, model_path)
    visualize_metrics(metrics, y_test, y_pred_proba)


if __name__ == "__main__":
    run_training_script(
        TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, METRICS_PATH, PARAMS_PATH
    )
