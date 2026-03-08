from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


JOB_YEAR_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

TARGET_MAP = {
    "Loan Given": 1,
    "Loan Refused": 0,
    "Loan Approved": 1,
    "Loan Rejected": 0,
}


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series | None


class LoanDataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans the raw loan dataset and adds engineered features.
    Important:
    - This class does NOT drop Loan Status / Loan ID / Customer ID
    - That is handled later in prepare_dataset()
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop duplicate loans, keeping first occurrence
        if "Loan ID" in df.columns:
            df = df.drop_duplicates(subset="Loan ID", keep="first")

        # Normalize categorical values
        if "Home Ownership" in df.columns:
            df["Home Ownership"] = (
                df["Home Ownership"]
                .replace(
                    {
                        "HaveMortgage": "Home Mortgage",
                        "Own Home": "Own",
                    }
                )
                .astype("object")
            )

        if "Purpose" in df.columns:
            df["Purpose"] = df["Purpose"].replace({"Other": "other"})

        # Clean mixed-format numeric text columns
        for col in ["Monthly Debt", "Maximum Open Credit"]:
            if col in df.columns:
                df[col] = self._to_numeric(df[col])

        # Force numeric conversion on known numeric columns
        numeric_cols = [
            "Current Loan Amount",
            "Credit Score",
            "Annual Income",
            "Monthly Debt",
            "Years of Credit History",
            "Months since last delinquent",
            "Number of Open Accounts",
            "Number of Credit Problems",
            "Current Credit Balance",
            "Maximum Open Credit",
            "Bankruptcies",
            "Tax Liens",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Replace fake placeholder values for loan amount
        if "Current Loan Amount" in df.columns:
            df["Current Loan Amount"] = df["Current Loan Amount"].replace(
                {99999999: np.nan, 9999999: np.nan}
            )

        # Fix inflated / invalid credit scores
        if "Credit Score" in df.columns:
            high_score_mask = df["Credit Score"] > 850
            df.loc[high_score_mask, "Credit Score"] = (
                df.loc[high_score_mask, "Credit Score"] / 10.0
            )

            invalid_score_mask = (df["Credit Score"] < 300) | (df["Credit Score"] > 850)
            df.loc[invalid_score_mask, "Credit Score"] = np.nan

        # Convert job tenure text to numeric
        if "Years in current job" in df.columns:
            df["Years in current job"] = df["Years in current job"].map(JOB_YEAR_MAP)

        # Feature engineering: debt-to-income ratio
        if {"Monthly Debt", "Annual Income"}.issubset(df.columns):
            monthly_income = df["Annual Income"] / 12.0
            df["Debt_to_Income_Ratio"] = df["Monthly Debt"] / monthly_income.replace(
                0, np.nan
            )

        # Feature engineering: credit utilization ratio
        if {"Current Credit Balance", "Maximum Open Credit"}.issubset(df.columns):
            df["Credit_Utilization_Ratio"] = df["Current Credit Balance"] / df[
                "Maximum Open Credit"
            ].replace(0, np.nan)

        # Feature engineering: delinquency history flag
        if "Months since last delinquent" in df.columns:
            df["Has_Delinquency_History"] = np.where(
                df["Months since last delinquent"].isna(), 0, 1
            )

        # Feature engineering: serious derogatory flag
        if {"Bankruptcies", "Tax Liens"}.issubset(df.columns):
            df["Serious_Derogatory_Flag"] = np.where(
                (df["Bankruptcies"].fillna(0) > 0) | (df["Tax Liens"].fillna(0) > 0),
                1,
                0,
            )

        return df

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        cleaned = (
            series.astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .str.replace("#VALUE!", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce")


def prepare_dataset(df: pd.DataFrame) -> DatasetBundle:
    """
    Prepare dataset for training or prediction.
    Clean first, then extract target so X and y stay aligned.
    """
    cleaner = LoanDataCleaner()
    cleaned_df = cleaner.fit_transform(df.copy())

    y = None
    if "Loan Status" in cleaned_df.columns:
        y = cleaned_df["Loan Status"].map(TARGET_MAP)

    X = cleaned_df.drop(
        columns=[
            c
            for c in ["Loan Status", "Loan ID", "Customer ID"]
            if c in cleaned_df.columns
        ],
        errors="ignore",
    )

    return DatasetBundle(X=X, y=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build sklearn preprocessing for numeric and categorical columns.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
