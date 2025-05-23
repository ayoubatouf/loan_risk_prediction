from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.utils.logger import training_logger


def evaluate(y_test, y_test_pred, y_test_prob):
    try:
        training_logger.info("Starting model evaluation.")

        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_prob)
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix,
        }

        training_logger.info(f"Evaluation metrics: {metrics}")
        training_logger.info("Model evaluation completed successfully.")

        return metrics

    except Exception as e:
        training_logger.error(f"Error during model evaluation: {e}", exc_info=True)
        raise
