import matplotlib.pyplot as plt
import seaborn as sns
from config.paths import *
from sklearn.feature_selection import mutual_info_classif
import pandas as pd


def plot_numeric_column_distributions(df, output_image_path):
    sns.set(style="whitegrid")

    ncols = 4
    nrows = (len(df.select_dtypes(include=["number"]).columns) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))

    axes = axes.flatten()

    for i, col in enumerate(df.select_dtypes(include=["number"]).columns):
        ax = axes[i]
        sns.histplot(df[col], kde=True, bins=10, color="skyblue", ax=ax)

        ax.set_title(f"{col}", fontsize=16)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    fig.savefig(output_image_path)


def plot_numeric_boxplots(df, output_image_path):
    sns.set(style="whitegrid")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    ncols = 4
    nrows = (len(numeric_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(x=df[col], color="skyblue", ax=ax)
        ax.set_title(f"{col}", fontsize=16)
        ax.set_xlabel(col, fontsize=12)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    fig.savefig(output_image_path)


def plot_correlation_heatmap(df, output_image_path, exclude_column="loan_status"):
    num_cols = df.select_dtypes(include="number").drop(
        columns=exclude_column, errors="ignore"
    )
    corr = num_cols.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
    )
    plt.title(f"Correlation Matrix (excluding {exclude_column})", fontsize=16)
    plt.tight_layout()

    plt.savefig(output_image_path)


def plot_mutual_information(df, output_image_path, target_column="loan_status"):
    X = df.drop(columns=target_column)
    y = df[target_column]

    mi_scores = mutual_info_classif(X, y, discrete_features="auto", random_state=0)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mi_series.plot(kind="barh", color="teal")
    plt.xlabel("Mutual Information Score")
    plt.title(f"Mutual Information with Target ({target_column})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(output_image_path)


if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    plot_numeric_column_distributions(
        raw_df, os.path.join(PLOTS_PATH, "numeric_distributions.png")
    )
    plot_numeric_boxplots(raw_df, os.path.join(PLOTS_PATH, "numeric_boxplots.png"))
    plot_correlation_heatmap(
        raw_df, os.path.join(PLOTS_PATH, "correlation_heatmap.png")
    )
    plot_mutual_information(full_df, os.path.join(PLOTS_PATH, "mutual_information.png"))
