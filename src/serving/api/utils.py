import os
import pandas as pd
import sqlite3
from datetime import datetime
from config.paths import *


def save_to_csv(input_data: dict, prediction: int):
    CSV_FILE_PATH = os.path.join(LOGS_PATH, "production_logs.csv")

    if not os.path.exists(CSV_FILE_PATH):
        df = pd.DataFrame(columns=list(input_data.keys()) + ["prediction", "timestamp"])
        df.to_csv(CSV_FILE_PATH, index=False)

    input_data["prediction"] = prediction
    input_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([input_data])
    df.to_csv(CSV_FILE_PATH, mode="a", header=False, index=False)


def save_to_db(input_data, prediction):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            person_age, person_income, person_home_ownership, person_emp_length,
            loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
            cb_person_default_on_file, cb_person_cred_hist_length, prediction, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            input_data["person_age"],
            input_data["person_income"],
            input_data["person_home_ownership"],
            input_data["person_emp_length"],
            input_data["loan_intent"],
            input_data["loan_grade"],
            input_data["loan_amnt"],
            input_data["loan_int_rate"],
            input_data["loan_percent_income"],
            input_data["cb_person_default_on_file"],
            input_data["cb_person_cred_hist_length"],
            prediction,
            datetime.now(),
        ),
    )
    conn.commit()
    conn.close()


def fetch_db_data(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")

    rows = cursor.fetchall()

    predictions = []
    for row in rows:
        prediction = {
            "id": row[0],
            "person_age": row[1],
            "person_income": row[2],
            "person_home_ownership": row[3],
            "person_emp_length": row[4],
            "loan_intent": row[5],
            "loan_grade": row[6],
            "loan_amnt": row[7],
            "loan_int_rate": row[8],
            "loan_percent_income": row[9],
            "cb_person_default_on_file": row[10],
            "cb_person_cred_hist_length": row[11],
            "prediction": row[12],
            "timestamp": row[13],
        }
        predictions.append(prediction)

    conn.close()
    return predictions
