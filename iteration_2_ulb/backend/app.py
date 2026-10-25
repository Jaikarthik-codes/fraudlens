import os
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, create_model
import pandas as pd

# Models, scaler and the shared scaling code live in scoring.py, which the
# Streamlit dashboard also uses
from scoring import INPUT_COLUMNS as REQUIRED_FIELDS, MODELS, MAIN_MODEL as model, prepare

app = FastAPI(title="FraudLens API")

# Allow web pages on other sites to call this API from the browser.
# Set CORS_ORIGINS="https://your-site.example" on Render to lock it down;
# the default "*" allows any website.
ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

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
    return prepare(pd.DataFrame([transaction.model_dump()]))


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
    for name, m in MODELS.items():
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
