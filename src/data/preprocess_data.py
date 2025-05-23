import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import pickle
from config.paths import *
from src.utils.logger import preprocessing_logger


def preprocess_data(df, mode="train"):
    try:
        preprocessing_logger.info(f"Starting data preprocessing in '{mode}' mode.")

        if mode == "train":
            df = df.drop_duplicates()
            df.loc[:, "person_emp_length"] = df["person_emp_length"].fillna(
                df["person_emp_length"].min()
            )
            df.loc[:, "loan_int_rate"] = df["loan_int_rate"].fillna(
                df["loan_int_rate"].mean()
            )
            preprocessing_logger.info(
                "Handled missing values for 'person_emp_length' and 'loan_int_rate'."
            )

        home_ownership_mapping = {"RENT": 2, "OWN": 3, "MORTGAGE": 4, "OTHER": 1}
        df.loc[:, "person_home_ownership"] = df["person_home_ownership"].map(
            home_ownership_mapping
        )
        preprocessing_logger.info("Mapped 'person_home_ownership' categories.")

        label_encoder = LabelEncoder()
        df.loc[:, "loan_grade"] = label_encoder.fit_transform(df["loan_grade"])
        preprocessing_logger.info("Encoded 'loan_grade'.")

        df.loc[:, "cb_person_default_on_file"] = df["cb_person_default_on_file"].map(
            {"N": 0, "Y": 1}
        )
        preprocessing_logger.info("Mapped 'cb_person_default_on_file' values.")

        df = pd.get_dummies(df, columns=["loan_intent"], drop_first=False)
        preprocessing_logger.info("Applied one-hot encoding to 'loan_intent'.")

        if mode == "train":
            features = [col for col in df.columns if col != "loan_status"]
            with open(FEATURES_PATH, "wb") as f:
                pickle.dump(features, f)
            preprocessing_logger.info(f"Saved feature list to {FEATURES_PATH}.")
            df = sample_data(df)

        elif mode == "inference":
            with open(FEATURES_PATH, "rb") as f:
                saved_columns = pickle.load(f)

            for col in saved_columns:
                if col not in df:
                    df[col] = 0
            df = df[saved_columns]
            preprocessing_logger.info(
                "Aligned inference data with training feature set."
            )

        preprocessing_logger.info("Data preprocessing completed successfully.")
        return df

    except Exception as e:
        preprocessing_logger.error(f"Error during preprocessing: {e}", exc_info=True)
        raise


def sample_data(df):
    try:
        preprocessing_logger.info("Starting data balancing with undersampling.")

        majority_class = df[df["loan_status"] == 0]
        minority_class = df[df["loan_status"] == 1]

        majority_class_undersampled = resample(
            majority_class,
            replace=False,
            n_samples=len(minority_class),
            random_state=42,
        )

        df_balanced = pd.concat([majority_class_undersampled, minority_class])
        df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        preprocessing_logger.info("Data balancing completed.")
        return df

    except Exception as e:
        preprocessing_logger.error(f"Error during data sampling: {e}", exc_info=True)
        raise


def split_and_scale(df):
    try:
        preprocessing_logger.info("Starting data split and scaling.")

        scaler = MinMaxScaler()
        X = df.drop("loan_status", axis=1)
        y = df["loan_status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        preprocessing_logger.info("Data split and scaling completed.")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    except Exception as e:
        preprocessing_logger.error(f"Error during split and scale: {e}", exc_info=True)
        raise
