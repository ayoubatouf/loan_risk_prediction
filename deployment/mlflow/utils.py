import os
import mlflow
import numpy as np
import json
import subprocess
import sys
import matplotlib.pyplot as plt
from config.paths import *


def log_metrics(metrics):
    artifact_metrics = {}
    for metric_name, metric_value in metrics.items():
        try:
            if (
                isinstance(metric_value, (np.ndarray, list))
                and np.size(metric_value) == 1
            ):
                metric_value = float(np.squeeze(metric_value))
            elif isinstance(metric_value, np.generic):
                metric_value = float(metric_value)

            mlflow.log_metric(metric_name, float(metric_value))
        except Exception:
            artifact_metrics[metric_name] = (
                metric_value.tolist()
                if hasattr(metric_value, "tolist")
                else str(metric_value)
            )
    if artifact_metrics:
        artifact_path = "mlflow_artifacts/complex_metrics.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact_metrics, f, indent=2)
        mlflow.log_artifact(artifact_path)


def save_samples(X_train, y_train, X_test, y_test):
    os.makedirs("mlflow_artifacts", exist_ok=True)

    sample_size = 100
    X_train.sample(min(sample_size, len(X_train))).to_csv(
        "mlflow_artifacts/X_train_sample.csv", index=False
    )
    y_train.sample(min(sample_size, len(y_train))).to_csv(
        "mlflow_artifacts/y_train_sample.csv", index=False
    )
    X_test.sample(min(sample_size, len(X_test))).to_csv(
        "mlflow_artifacts/X_test_sample.csv", index=False
    )
    y_test.sample(min(sample_size, len(y_test))).to_csv(
        "mlflow_artifacts/y_test_sample.csv", index=False
    )
    mlflow.log_artifacts("mlflow_artifacts")


def plot_and_log_confusion_matrix(cm):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(cax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("mlflow_artifacts/confusion_matrix.png")
    plt.close(fig)
    mlflow.log_artifact("mlflow_artifacts/confusion_matrix.png")


def log_requirements():
    with open("mlflow_artifacts/requirements.txt", "w") as f:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    mlflow.log_artifact("mlflow_artifacts/requirements.txt")
