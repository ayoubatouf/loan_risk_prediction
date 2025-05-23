import random


def generate_fake_data():
    return {
        "person_age": random.uniform(20, 60),
        "person_income": random.uniform(30000, 150000),
        "person_home_ownership": random.choice(["RENT", "MORTGAGE", "OWN", "OTHER"]),
        "person_emp_length": random.uniform(0, 40),
        "loan_intent": random.choice(
            [
                "EDUCATION",
                "MEDICAL",
                "VENTURE",
                "PERSONAL",
                "DEBTCONSOLIDATION",
                "HOMEIMPROVEMENT",
            ]
        ),
        "loan_grade": random.choice(["A", "B", "C", "D", "E", "F", "G"]),
        "loan_amnt": random.uniform(1000, 50000),
        "loan_int_rate": random.uniform(5, 25),
        "loan_percent_income": random.uniform(5, 50),
        "cb_person_default_on_file": random.choice(["Y", "N"]),
        "cb_person_cred_hist_length": random.uniform(0, 15),
    }
