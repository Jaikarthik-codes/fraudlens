from pathlib import Path
import os
from typing import Annotated

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field, create_model
import joblib
import numpy as np

app = FastAPI(title="FraudLens API")

# Allow the React app (Vercel / Vite dev server) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Models live next to the backend folder; resolve from this file so the app
# works no matter which directory it is started from
MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'

# Load the trained model
model = joblib.load(MODELS_DIR / 'rf_model.pkl')

# Scaler fitted on raw Amount and Time (see notebooks/02_preprocess.py)
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

# Order matters: this is the column order the model was trained on
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


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# blocking scikit-learn call does not freeze the event loop.
@app.post('/predict')
def predict(transaction: Transaction):
    data = transaction.model_dump()

    # Scale Amount and Time the same way as during training
    scaled_amount, scaled_time = scaler.transform([[data['Amount'], data['Time']]])[0]

    # Build feature array in correct order
    features = [data[f] for f in V_FEATURES] + [scaled_amount, scaled_time]

    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]
    result = "FRAUD" if prediction == 1 else "LEGITIMATE"

    return {
        'prediction': result.lower(),
        'confidence': round(float(probability), 4)
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
