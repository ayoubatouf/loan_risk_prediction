from src.models.evaluate import evaluate
from src.models.predict import predict
from src.models.train import train
from src.utils.logger import training_logger


def run_training_pipeline(X_train, X_test, y_train, y_test, best_params):
    try:
        training_logger.info("Starting the training pipeline.")

        model = train(X_train, y_train, best_params)
        training_logger.info("Model training completed successfully.")

        y_pred, y_pred_proba = predict(model, X_test)
        training_logger.info("Prediction completed successfully.")

        metrics = evaluate(y_test, y_pred, y_pred_proba)
        training_logger.info(f"Model evaluation metrics: {metrics}")

        training_logger.info("Training pipeline finished successfully.")

        return model, metrics, y_pred, y_pred_proba

    except Exception as e:
        training_logger.error(f"Error during the training pipeline: {e}", exc_info=True)
        raise
