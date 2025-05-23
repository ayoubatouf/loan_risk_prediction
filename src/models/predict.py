from src.utils.logger import inference_logger


def predict(model, X_test):
    try:
        inference_logger.info("Starting prediction on data.")

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        inference_logger.info("Prediction completed successfully.")
        return y_pred, y_pred_proba

    except Exception as e:
        inference_logger.error(f"Error during prediction: {e}", exc_info=True)
        raise
