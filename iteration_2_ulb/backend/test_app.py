import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app, REQUIRED_FIELDS

# The five demo transactions shown in the React page. Their expected verdicts
# were checked against the original Flask app and match NOTES.md (4 of 5 fraud).
SAMPLES = json.loads((Path(__file__).parent / 'sample_transactions.json').read_text())


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize('sample', SAMPLES, ids=[s['label'][:20] for s in SAMPLES])
def test_demo_transactions_keep_their_verdicts(client, sample):
    response = client.post('/predict', json=sample['data'])
    assert response.status_code == 200
    assert response.json()['prediction'] == sample['expected_random_forest']


def test_compare_returns_all_models_and_ensembles(client):
    response = client.post('/compare', json=SAMPLES[1]['data'])
    assert response.status_code == 200
    body = response.json()
    assert set(body['models']) == {'random_forest', 'gradient_boosting', 'adaboost'}
    assert 0 <= body['hard_voting']['votes_for_fraud'] <= 3
    assert 0.0 <= body['soft_voting']['confidence'] <= 1.0


def test_compare_ensembles_follow_their_rules(client):
    body = client.post('/compare', json=SAMPLES[1]['data']).json()
    scores = [m['confidence'] for m in body['models'].values()]
    votes = sum(m['prediction'] == 'fraud' for m in body['models'].values())
    assert body['hard_voting']['votes_for_fraud'] == votes
    assert (body['hard_voting']['prediction'] == 'fraud') == (votes >= 2)
    # soft voting averages the three scores (allow for 4-decimal rounding)
    assert body['soft_voting']['confidence'] == pytest.approx(sum(scores) / 3, abs=1e-3)


def test_compare_rejects_bad_input(client):
    assert client.post('/compare', json={'V1': 1.0}).status_code == 422


def test_random_forest_in_compare_matches_predict(client):
    data = SAMPLES[0]['data']
    single = client.post('/predict', json=data).json()
    rf = client.post('/compare', json=data).json()['models']['random_forest']
    assert rf == single


def valid_payload():
    return {field: 0.0 for field in REQUIRED_FIELDS}


def bad_fields(response):
    # FastAPI's 422 body lists each error with a location like ["body", "V1"]
    return sorted(error['loc'][-1] for error in response.json()['detail'])


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post('/predict', json=valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body['prediction'] in ('fraud', 'legitimate')
    assert 0.0 <= body['confidence'] <= 1.0


def test_predict_accepts_integers(client):
    payload = valid_payload()
    payload['Time'] = 406
    assert client.post('/predict', json=payload).status_code == 200


def test_predict_missing_fields_returns_422(client):
    payload = valid_payload()
    del payload['V1']
    del payload['Amount']
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
    assert bad_fields(response) == ['Amount', 'V1']


def test_predict_non_numeric_field_returns_422(client):
    payload = valid_payload()
    payload['V3'] = 'abc'
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
    assert bad_fields(response) == ['V3']


def test_predict_non_json_body_returns_422(client):
    response = client.post('/predict', content='not json', headers={'Content-Type': 'text/plain'})
    assert response.status_code == 422
