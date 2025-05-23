import lightgbm as lgb
from src.utils.logger import training_logger


def train(X_train, y_train, best_params):
    try:
        training_logger.info("Starting model training with LightGBM.")
        training_logger.info(f"Training parameters: {best_params}")

        best_params["verbose"] = -1
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)

        training_logger.info("Model training completed successfully.")
        return model

    except Exception as e:
        training_logger.error(f"Error during model training: {e}", exc_info=True)
        raise
