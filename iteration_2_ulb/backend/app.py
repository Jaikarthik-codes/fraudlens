from pathlib import Path
import os
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, create_model
import joblib
import pandas as pd

app = FastAPI(title="FraudLens API")

# Allow the React app (Vercel / Vite dev server) to call this API.
# Set CORS_ORIGINS="https://your-site.vercel.app" on Render to lock it down;
# the default "*" allows any website.
ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

# Models live next to the backend folder; resolve from this file so the app
# works no matter which directory it is started from
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Load the trained models. Random Forest is the main model (best F1);
# the other two are only used by /compare.
model = joblib.load(MODELS_DIR / 'rf_model.pkl')
COMPARE_MODELS = {
    'random_forest': model,
    'gradient_boosting': joblib.load(MODELS_DIR / 'gb_model.pkl'),
    'adaboost': joblib.load(MODELS_DIR / 'ada_model.pkl'),
}

# Scaler fitted on raw Amount and Time (see notebooks/02_preprocess.py)
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

# The 28 anonymised PCA columns from the ULB dataset
V_FEATURES = [f'V{i}' for i in range(1, 29)]
REQUIRED_FIELDS = V_FEATURES + ['Amount', 'Time']

# Request schema: 30 required numbers. FastAPI/Pydantic rejects missing or
# non-numeric fields with a 422 response that names the bad fields, so no
# manual validation is needed. strict=True stops strings like "1.5" or true.
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Transaction = create_model('Transaction', **{name: (Number, ...) for name in REQUIRED_FIELDS})


@app.get('/health')
def health():
    return {'status': 'ok'}


def build_features(transaction):
    """Turn a validated request into the one-row table the models were trained on."""
    data = transaction.model_dump()

    # Scale Amount and Time the same way as during training
    scaled_amount, scaled_time = scaler.transform(
        pd.DataFrame([[data['Amount'], data['Time']]], columns=['Amount', 'Time'])
    )[0]

    row = {name: data[name] for name in V_FEATURES}
    row['scaled_Amount'] = scaled_amount
    row['scaled_Time'] = scaled_time

    # The models remember the column names and order they were trained on.
    # Using that order (and passing named columns) means scikit-learn itself
    # raises an error if a column is ever missing or misnamed.
    return pd.DataFrame([row], columns=list(model.feature_names_in_))


def label(is_fraud):
    return 'fraud' if is_fraud else 'legitimate'


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# blocking scikit-learn call does not freeze the event loop.
@app.post('/predict')
def predict(transaction: Transaction):
    features_array = build_features(transaction)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]

    return {
        'prediction': label(prediction == 1),
        'confidence': round(float(probability), 4)
    }


@app.post('/compare')
def compare(transaction: Transaction):
    """Score one transaction with all three models and both ensembles.

    Mirrors notebooks/04_ensemble.py: hard voting needs at least 2 of the
    3 models to say fraud; soft voting averages the 3 fraud probabilities
    and flags fraud at 0.5 or above.
    """
    features_array = build_features(transaction)

    models = {}
    probabilities = []
    votes = 0
    for name, m in COMPARE_MODELS.items():
        probability = float(m.predict_proba(features_array)[0][1])
        is_fraud = m.predict(features_array)[0] == 1
        probabilities.append(probability)
        votes += int(is_fraud)
        models[name] = {'prediction': label(is_fraud), 'confidence': round(probability, 4)}

    average = sum(probabilities) / len(probabilities)
    return {
        'models': models,
        'hard_voting': {'prediction': label(votes >= 2), 'votes_for_fraud': votes},
        'soft_voting': {'prediction': label(average >= 0.5), 'confidence': round(average, 4)},
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
