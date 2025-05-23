from src.pipelines.inference_pipeline import run_inference_pipeline
from config.paths import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from src.serving.api.utils import save_to_csv, save_to_db
import pandas as pd

app = FastAPI()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class InputData(BaseModel):
    person_age: float
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: float


@app.post("/predict")
def predict_route(input_data: InputData):
    try:
        input_df = pd.DataFrame([input_data.dict()])

        y_pred, _ = run_inference_pipeline(model, scaler, input_df)
        save_to_csv(input_data.dict(), int(y_pred[0]))
        save_to_db(input_data.dict(), int(y_pred[0]))

        return {
            "prediction": y_pred.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
