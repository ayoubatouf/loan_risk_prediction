import os
from config.paths import *
import joblib
from src.pipelines.data_pipeline import run_data_pipeline


def run_data_script(raw_path, train_path, test_path, scaler_path):

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = run_data_pipeline(raw_path)

    train_data = X_train.copy()
    train_data["loan_status"] = y_train.reset_index(drop=True)

    test_data = X_test.copy()
    test_data["loan_status"] = y_test.reset_index(drop=True)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)


if __name__ == "__main__":

    run_data_script(RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, SCALER_PATH)
