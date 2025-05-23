from src.data.preprocess_data import preprocess_data
from src.models.predict import predict
from src.utils.logger import inference_logger


def run_inference_pipeline(model, scaler, data):
    try:
        inference_logger.info("Starting inference pipeline.")

        data = preprocess_data(data, mode="inference")
        inference_logger.info("Data preprocessing completed.")

        data = scaler.transform(data)
        inference_logger.info("Data scaling completed.")

        y_pred, y_pred_proba = predict(model, data)
        inference_logger.info("Inference completed successfully.")

        return y_pred, y_pred_proba

    except Exception as e:
        inference_logger.error(f"Error during inference pipeline: {e}", exc_info=True)
        raise
