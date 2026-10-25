import pytest
from fastapi.testclient import TestClient

from app import app, REQUIRED_FIELDS


@pytest.fixture
def client():
    return TestClient(app)


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
