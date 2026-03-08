from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import METRICS_FILE, MODEL_FILE, RANDOM_STATE, TEST_SIZE, TRAIN_FILE
from .preprocessing import build_preprocessor, prepare_dataset


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'classification_report': classification_report(y_test, y_pred),
    }


def train() -> None:
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    bundle = prepare_dataset(df)
    X, y = bundle.X, bundle.y
    if y is None:
        raise ValueError('Training target was not found.')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    candidates = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    best_name = None
    best_model = None
    best_score = -1.0

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics

        if metrics['roc_auc'] > best_score:
            best_score = metrics['roc_auc']
            best_name = name
            best_model = pipeline

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)

    payload = {
        'best_model': best_name,
        'best_score_metric': 'roc_auc',
        'models': results,
    }
    METRICS_FILE.write_text(json.dumps(payload, indent=2))

    print(f'Saved model to: {MODEL_FILE}')
    print(f'Saved metrics to: {METRICS_FILE}')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    train()
