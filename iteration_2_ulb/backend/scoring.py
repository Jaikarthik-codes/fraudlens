"""
Shared scoring code for FraudLens.

Both the FastAPI backend (app.py) and the Streamlit dashboard use this file,
so a transaction is always scaled and arranged in exactly the same way.

The steps match the training notebooks:
  1. scale Amount and Time with the saved scaler (models/scaler.pkl)
  2. put the 30 columns in the order the models were trained on
  3. ask each model for its fraud probability
"""
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# The 28 anonymised PCA columns from the ULB dataset, plus the two raw columns
V_FEATURES = [f'V{i}' for i in range(1, 29)]
INPUT_COLUMNS = V_FEATURES + ['Amount', 'Time']

# Scaler fitted on raw Amount and Time (see notebooks/02_preprocess.py)
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

# Random Forest is the main model (best F1); the other two are for comparison
MODELS = {
    'random_forest': joblib.load(MODELS_DIR / 'rf_model.pkl'),
    'gradient_boosting': joblib.load(MODELS_DIR / 'gb_model.pkl'),
    'adaboost': joblib.load(MODELS_DIR / 'ada_model.pkl'),
}
MAIN_MODEL = MODELS['random_forest']

MODEL_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'adaboost': 'AdaBoost',
}


def prepare(transactions):
    """Turn raw transactions (V1-V28, Amount, Time) into the table the models expect."""
    missing = [c for c in INPUT_COLUMNS if c not in transactions.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    features = transactions[V_FEATURES].astype(float).copy()

    # Scale Amount and Time the same way as during training
    scaled = scaler.transform(transactions[['Amount', 'Time']].astype(float))
    features['scaled_Amount'] = scaled[:, 0]
    features['scaled_Time'] = scaled[:, 1]

    # The models remember the column names and order they were trained on.
    # Using that order (and passing named columns) means scikit-learn itself
    # raises an error if a column is ever missing or misnamed.
    return features[list(MAIN_MODEL.feature_names_in_)]


def fraud_scores(transactions):
    """Fraud probability (0 to 1) from each model, plus the soft-voting average.

    Returns one row per transaction with the columns random_forest,
    gradient_boosting, adaboost and soft_voting.
    """
    features = prepare(transactions)
    scores = pd.DataFrame(
        {name: model.predict_proba(features)[:, 1] for name, model in MODELS.items()},
        index=transactions.index,
    )
    # Soft voting = average of the three probabilities (as in 04_ensemble.py)
    scores['soft_voting'] = scores[list(MODELS)].mean(axis=1)
    return scores
