import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union

from src.config import MODEL_FILE
from src.preprocessing import prepare_dataset

app = FastAPI(title="Loan Eligibility API", version="1.0.0")
model = joblib.load(MODEL_FILE) if MODEL_FILE.exists() else None


class LoanApplication(BaseModel):
    loan_id: Union[str, int]
    customer_id: Union[str, int]
    current_loan_amount: Optional[float] = None
    term: Optional[str] = None
    credit_score: Optional[float] = None
    years_in_current_job: Optional[str] = None
    home_ownership: Optional[str] = None
    annual_income: Optional[float] = None
    purpose: Optional[str] = None
    monthly_debt: Optional[float] = None
    years_of_credit_history: Optional[float] = None
    months_since_last_delinquent: Optional[float] = None
    number_of_open_accounts: Optional[float] = None
    number_of_credit_problems: Optional[float] = None
    current_credit_balance: Optional[float] = None
    maximum_open_credit: Optional[float] = None
    bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Loan ID": self.loan_id,
                    "Customer ID": self.customer_id,
                    "Current Loan Amount": self.current_loan_amount,
                    "Term": self.term,
                    "Credit Score": self.credit_score,
                    "Years in current job": self.years_in_current_job,
                    "Home Ownership": self.home_ownership,
                    "Annual Income": self.annual_income,
                    "Purpose": self.purpose,
                    "Monthly Debt": self.monthly_debt,
                    "Years of Credit History": self.years_of_credit_history,
                    "Months since last delinquent": self.months_since_last_delinquent,
                    "Number of Open Accounts": self.number_of_open_accounts,
                    "Number of Credit Problems": self.number_of_credit_problems,
                    "Current Credit Balance": self.current_credit_balance,
                    "Maximum Open Credit": self.maximum_open_credit,
                    "Bankruptcies": self.bankruptcies,
                    "Tax Liens": self.tax_liens,
                }
            ]
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(application: LoanApplication) -> dict:
    if model is None:
        return {"error": "Model file not found. Train the model first."}

    raw_df = application.to_frame()
    bundle = prepare_dataset(raw_df)
    X = bundle.X

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    return {
        "prediction": "Loan Given" if pred == 1 else "Loan Refused",
        "approval_probability": proba,
    }
