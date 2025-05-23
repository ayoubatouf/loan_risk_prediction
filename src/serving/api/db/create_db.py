import sqlite3
from config.paths import DB_PATH


def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_age REAL,
            person_income REAL,
            person_home_ownership TEXT,
            person_emp_length REAL,
            loan_intent TEXT,
            loan_grade TEXT,
            loan_amnt REAL,
            loan_int_rate REAL,
            loan_percent_income REAL,
            cb_person_default_on_file TEXT,
            cb_person_cred_hist_length REAL,
            prediction INTEGER,
            timestamp DATETIME
        )
    """
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_table()
