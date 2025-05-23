import os
import pandas as pd
from src.utils.data_drift import has_drift
from config.paths import *

if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_PATH)
    production_df = pd.read_csv(os.path.join(LOGS_PATH, "production_logs.csv"))

    raw_df = raw_df.drop(columns=["loan_status"], errors="ignore")
    production_df = production_df.drop(columns=["prediction"], errors="ignore")
    columns_to_check = raw_df.columns.tolist()

    drift_flag, drifted_columns = has_drift(raw_df, production_df, columns_to_check)

    print("\nDrift detected:", drift_flag)
    print("Drift found in columns:", drifted_columns)
