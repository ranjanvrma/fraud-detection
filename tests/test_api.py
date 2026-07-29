from fastapi.testclient import TestClient

from api.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_expected_shape():
    payload = {
        "step": 2,
        "type": "TRANSFER",
        "amount": 9000,
        "sender_balance": 9500,
        "receiver_balance": 0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert 0.0 <= body["risk_score"] <= 100.0
    assert body["risk_level"] in {"LOW RISK", "MEDIUM RISK", "HIGH RISK"}
    assert isinstance(body["reasons"], list) and len(body["reasons"]) > 0


def test_predict_rejects_invalid_type():
    payload = {
        "step": 2,
        "type": "NOT_A_TYPE",
        "amount": 100,
        "sender_balance": 1000,
        "receiver_balance": 0,
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
