from src.data.loading_data import load_data
from src.data.preprocess_data import preprocess_data, split_and_scale
from src.utils.logger import preprocessing_logger


def run_data_pipeline(data_path):
    try:
        preprocessing_logger.info(f"Starting data pipeline with input: {data_path}")

        df = load_data(data_path)
        preprocessing_logger.info("Data loaded successfully.")

        df = preprocess_data(df)
        preprocessing_logger.info("Data preprocessing completed.")

        X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
        preprocessing_logger.info("Data split and scaling completed.")

        preprocessing_logger.info("Data pipeline finished successfully.")
        return X_train, X_test, y_train, y_test, scaler

    except Exception as e:
        preprocessing_logger.error(f"Error in data pipeline: {e}", exc_info=True)
        raise
