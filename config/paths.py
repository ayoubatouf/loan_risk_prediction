import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_DATA_PATH = os.path.join(PROJECT_PATH, "data", "raw", "credit_risk_dataset.csv")
TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed", "train.csv")
TEST_DATA_PATH = os.path.join(PROJECT_PATH, "data", "processed", "test.csv")
SCALER_PATH = os.path.join(PROJECT_PATH, "src", "features", "scaler.pkl")
MODEL_PATH = os.path.join(PROJECT_PATH, "results", "pretrained_models", "model.pkl")
METRICS_PATH = os.path.join(PROJECT_PATH, "results", "evaluation_results")
PARAMS_PATH = os.path.join(PROJECT_PATH, "src", "models", "best_params.json")
INFERENCE_DATA_PATH = os.path.join(
    PROJECT_PATH, "data", "inference", "inference_input.csv"
)
INFERENCE_RESULTS_PATH = os.path.join(
    PROJECT_PATH, "results", "inference_results", "inference_output.csv"
)
FEATURES_PATH = os.path.join(PROJECT_PATH, "src", "features", "features.pkl")
LOGS_PATH = os.path.join(PROJECT_PATH, "logs")
PLOTS_PATH = os.path.join(PROJECT_PATH, "reports", "figures")
DB_PATH = os.path.join(PROJECT_PATH, "src", "serving", "api", "db", "predictions.db")
