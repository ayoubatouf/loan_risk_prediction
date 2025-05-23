from deployment.mlflow.utils import *
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import time
from config.paths import *
from src.pipelines.training_pipeline import run_training_pipeline
from mlflow.models.signature import infer_signature


def main():
    train = pd.read_csv(TRAIN_DATA_PATH)
    X_train = train.drop(columns=["loan_status"])
    y_train = train["loan_status"]

    test = pd.read_csv(TEST_DATA_PATH)
    X_test = test.drop(columns=["loan_status"])
    y_test = test["loan_status"]

    with open(PARAMS_PATH, "r") as file:
        best_params = json.load(file)

    os.makedirs("mlflow_artifacts", exist_ok=True)

    start_time = time.time()

    with mlflow.start_run() as run:
        model, metrics, y_pred, y_pred_proba = run_training_pipeline(
            X_train, X_test, y_train, y_test, best_params
        )

        for param, value in best_params.items():
            mlflow.log_param(param, value)

        mlflow.set_tag("user", "ayoub")
        mlflow.set_tag("dataset", "loan_status_data_v1")
        mlflow.set_tag("model_description", "LightGBM model with tuned hyperparameters")

        log_metrics(metrics)
        save_samples(X_train, y_train, X_test, y_test)

        preds_df = pd.DataFrame(
            {"y_true": y_test, "y_pred": y_pred, "y_pred_proba": y_pred_proba}
        )
        preds_path = "mlflow_artifacts/predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        mlflow.log_artifact(preds_path)

        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            plot_and_log_confusion_matrix(cm)

        log_requirements()

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test.sample(1),
            signature=signature,
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model_name = "LoanStatusModel"
        model_version = mlflow.register_model(model_uri, registered_model_name)

        duration = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", duration)

        print("MLflow tracking complete. Run ID:", run.info.run_id)
        print(
            f"Model registered as '{registered_model_name}', version {model_version.version}"
        )


if __name__ == "__main__":
    main()
