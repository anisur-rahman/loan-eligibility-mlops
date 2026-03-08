from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_FILE, PREDICTIONS_DIR, TEST_FILE
from .preprocessing import prepare_dataset

LABEL_MAP = {1: "Loan Given", 0: "Loan Refused"}


def predict(test_file: Path = TEST_FILE, output_file: Path | None = None) -> Path:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run training first."
        )

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(test_file, low_memory=False)

    # Preserve identifiers for output
    ids = (
        df[["Loan ID", "Customer ID"]].copy()
        if {"Loan ID", "Customer ID"}.issubset(df.columns)
        else None
    )

    # Clean/prepare the test data the same way as training
    bundle = prepare_dataset(df)
    X = bundle.X

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    output = pd.DataFrame(
        {
            "prediction": [LABEL_MAP.get(int(v), str(v)) for v in predictions],
            "approval_probability": probabilities,
        }
    )

    if ids is not None:
        # align ids to cleaned rows in case duplicates were dropped
        cleaned_ids = (
            df.drop_duplicates(subset="Loan ID", keep="first")[
                ["Loan ID", "Customer ID"]
            ].reset_index(drop=True)
            if "Loan ID" in df.columns
            else ids.reset_index(drop=True)
        )
        output = pd.concat([cleaned_ids, output.reset_index(drop=True)], axis=1)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = output_file or (PREDICTIONS_DIR / "predictions.csv")
    output.to_csv(output_file, index=False)

    print(f"Saved predictions to: {output_file}")
    return output_file


if __name__ == "__main__":
    predict()
