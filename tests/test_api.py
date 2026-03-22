from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))

from app import app  # noqa: E402


SAMPLE_PAYLOAD = {
    "transaction_id": "tx-1001",
    "terminal_name": "STARBUCKS",
    "terminal_description": "COFFEE SHOP",
    "city": "Moscow",
    "amount": 450.0,
    "items": [
        {"name": "latte", "price": 250.0},
        {"name": "croissant", "price": 200.0},
    ],
}


def test_health():
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    client = app.test_client()
    response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200
    body = response.get_json()
    assert set(body.keys()) == {"prediction", "confidence"}
    assert isinstance(body["prediction"], int)
    assert 0.0 <= body["confidence"] <= 1.0
