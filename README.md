sour# Loan Eligibility MLOps Project

This is a cleaned, production-style rebuild of the uploaded loan approval project.
It keeps the same business goal: predict whether a loan should be granted based on applicant financial data. The original problem statement defines this as a supervised prediction task and sets a minimum target of 70% accuracy. fileciteturn0file0

## Project layout

```text
loan_eligibility_mlops/
├── app/
│   └── main.py
├── data/
│   ├── predictions/
│   └── raw/
├── models/
├── src/
│   ├── config.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   └── test_preprocessing.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## What changed from the notebook version

- moved logic out of notebooks into reusable Python modules
- replaced fragile `factorize()` encoding with `OneHotEncoder(handle_unknown="ignore")`
- packaged preprocessing and model together in a single sklearn `Pipeline`
- added engineered features such as debt-to-income ratio and credit utilization ratio
- added a FastAPI app for serving predictions
- added a basic pytest unit test
- added a Dockerfile for deployment

## Setup

```bash
cd loan_eligibility_mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Put the uploaded CSV files here

```text
data/raw/LoansTrainingSetV2.csv
data/raw/test_data.csv
```

## Train the model

```bash
PYTHONPATH=. python -m src.train
```

This writes:

- `models/loan_eligibility_pipeline.joblib`
- `models/metrics.json`

## Generate predictions

```bash
PYTHONPATH=. python -m src.predict
```

This writes:

- `data/predictions/predictions.csv`

## Run the API

```bash
PYTHONPATH=. uvicorn app.main:app --reload
```

### Sample request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id": "12345",
    "customer_id": "98765",
    "current_loan_amount": 15000,
    "term": "Short Term",
    "credit_score": 720,
    "years_in_current_job": "5 years",
    "home_ownership": "Rent",
    "annual_income": 85000,
    "purpose": "Debt Consolidation",
    "monthly_debt": "950.25",
    "years_of_credit_history": 12.5,
    "months_since_last_delinquent": 18,
    "number_of_open_accounts": 8,
    "number_of_credit_problems": 0,
    "current_credit_balance": 6200,
    "maximum_open_credit": "18000",
    "bankruptcies": 0,
    "tax_liens": 0
  }'
```
## Open the API in your browser
http://127.0.0.1:8000/health
You should see JSON like: {"status":"ok","model_loaded":true}

Then open: http://127.0.0.1:8000/docs
This opens the FastAPI interactive docs page where you can test /predict.
Test /predict from the docs page
click POST /predict
click Try it out
paste a JSON example

## FastAPI app Screenshots

![ML Pipeline](images/bank_loan-02.png)
![ML Pipeline](images/bank_loan-01.png)
![ML Pipeline](images/bank_loan-03.png)
![ML Pipeline](images/bank_loan-04.png)
![ML Pipeline](images/bank_loan-05.png)

## Notes

This rebuild is designed to be safer and more maintainable than the original notebooks. It does not try to exactly reproduce the notebook's SoftImpute workflow. Instead, it uses a robust production baseline with sklearn-native preprocessing and a reusable pipeline.
