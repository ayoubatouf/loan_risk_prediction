#!/bin/bash

curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "person_age": 22,
  "person_income": 59000,
  "person_home_ownership": "RENT",
  "person_emp_length": 123.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "D",
  "loan_amnt": 35000,
  "loan_int_rate": 16.02,
  "loan_percent_income": 0.59,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 3
}'
