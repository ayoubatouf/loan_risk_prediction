import os
import json
import pandas as pd
from pathlib import Path
from src.utils.logger import preprocessing_logger

SCHEMA_FILE = str((Path(__file__).parent / "schema.json").resolve())


def load_expected_schema(schema_path=SCHEMA_FILE):
    if not os.path.exists(schema_path):
        preprocessing_logger.error(f"Schema file not found: {schema_path}")
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = json.load(f)

    preprocessing_logger.info("Expected schema loaded successfully.")
    return schema


def validate_data(file_path, schema_path=SCHEMA_FILE):
    expected_schema = load_expected_schema(schema_path)

    if not os.path.exists(file_path):
        preprocessing_logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path, nrows=10, low_memory=False)
        preprocessing_logger.info(
            f"File preview loaded for validation from {file_path}"
        )
    except Exception as e:
        preprocessing_logger.error(f"Error reading file {file_path}: {e}")
        raise ValueError(f"Error reading file: {e}")

    errors = []

    if set(df.columns) != set(expected_schema.keys()):
        errors.append(
            f"Column mismatch. Expected: {set(expected_schema.keys())}, Found: {set(df.columns)}"
        )

    for col, expected_dtype in expected_schema.items():
        try:
            df[col].astype(expected_dtype)
        except ValueError:
            errors.append(
                f"Column '{col}' expected type {expected_dtype}, but found {df[col].dtype}"
            )

    if errors:
        for error in errors:
            preprocessing_logger.error(f"Validation Error: {error}")
        raise ValueError("Data validation failed. Please fix the above issues.")

    preprocessing_logger.info("Data validation passed.")
    return True


def load_data(file_path, schema_path=SCHEMA_FILE):
    try:
        if validate_data(file_path, schema_path):
            df = pd.read_csv(file_path, low_memory=False)
            preprocessing_logger.info(f"Data loaded successfully from {file_path}.")
            return df
    except Exception as e:
        preprocessing_logger.error(f"Failed to load data from {file_path}: {e}")
        raise
