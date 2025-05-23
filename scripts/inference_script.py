import os
import joblib
import pandas as pd
from config.paths import *
from src.pipelines.inference_pipeline import run_inference_pipeline


def run_inference_script(model_path, scaler_path, data_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    data = pd.read_csv(data_path)
    y_pred, y_pred_proba = run_inference_pipeline(model, scaler, data)
    data["predictions"] = y_pred
    data["predictions_proba"] = y_pred_proba
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    run_inference_script(
        MODEL_PATH, SCALER_PATH, INFERENCE_DATA_PATH, INFERENCE_RESULTS_PATH
    )
