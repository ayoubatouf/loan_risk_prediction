import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from config.paths import *


def visualize_metrics(metrics, y_test, y_test_prob, filename=METRICS_PATH):
    try:
        os.makedirs(filename, exist_ok=True)

        conf_matrix = metrics.pop("confusion_matrix")
        scalar_metrics = metrics.copy()

        fig, ax = plt.subplots(figsize=(10, 5))
        metric_names = list(scalar_metrics.keys())
        metric_values = list(scalar_metrics.values())

        metrics_df = pd.DataFrame({"Metric": metric_names, "Value": metric_values})

        sns.barplot(
            data=metrics_df,
            x="Metric",
            y="Value",
            hue="Metric",
            palette="coolwarm",
            ax=ax,
            legend=False,
        )
        ax.set_title("Evaluation Metrics")
        ax.set_ylim(0, 1)
        for i, v in enumerate(metric_values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(filename, "evaluation_metrics.png"), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(filename, "confusion_matrix.png"), dpi=300)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(filename, "roc_curve.png"), dpi=300)
        plt.close(fig)

    except Exception as e:
        raise e
