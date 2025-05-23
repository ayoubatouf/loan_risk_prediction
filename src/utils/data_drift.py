import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def has_drift(df1, df2, columns, alpha=0.05):
    drift_detected = False
    drift_columns = []

    for column in columns:
        if column not in df1.columns or column not in df2.columns:
            raise ValueError(f"Column '{column}' is not present in both datasets")

        col1 = df1[column].dropna()
        col2 = df2[column].dropna()

        if col1.dtype == "object":
            categories = set(col1.unique()).union(col2.unique())
            freq1 = col1.value_counts(normalize=True).reindex(categories, fill_value=0)
            freq2 = col2.value_counts(normalize=True).reindex(categories, fill_value=0)
            contingency_table = pd.DataFrame([freq1, freq2])
            _, p_value, _, _ = chi2_contingency(contingency_table)

        elif pd.api.types.is_numeric_dtype(col1):
            _, p_value = ks_2samp(col1, col2)

        else:
            continue

        if p_value < alpha:
            drift_detected = True
            drift_columns.append(column)

    return drift_detected, drift_columns
