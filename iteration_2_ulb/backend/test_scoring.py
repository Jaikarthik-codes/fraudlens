import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from scoring import fraud_scores, prepare

SAMPLES = json.loads((Path(__file__).parent / 'sample_transactions.json').read_text())
TABLE = pd.DataFrame([s['data'] for s in SAMPLES])


def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match='Amount'):
        prepare(TABLE.drop(columns=['Amount']))


def test_prepare_returns_the_training_column_order():
    columns = list(prepare(TABLE).columns)
    assert columns[:28] == [f'V{i}' for i in range(1, 29)]
    assert columns[28:] == ['scaled_Amount', 'scaled_Time']


def test_fraud_scores_has_one_row_per_transaction():
    scores = fraud_scores(TABLE)
    assert list(scores.columns) == ['random_forest', 'gradient_boosting', 'adaboost', 'soft_voting']
    assert len(scores) == len(TABLE)
    assert scores.to_numpy().min() >= 0 and scores.to_numpy().max() <= 1


def test_batch_scores_match_the_api():
    # The dashboard scores many rows at once; the API scores one. They must agree.
    scores = fraud_scores(TABLE)
    client = TestClient(app)
    for i, sample in enumerate(SAMPLES):
        api = client.post('/predict', json=sample['data']).json()
        assert round(scores['random_forest'].iloc[i], 4) == api['confidence']
