from mcc_classifier.api.app import app


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


def create_client():
    return app.test_client()


def test_health():
    client = create_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    client = create_client()
    response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200
    body = response.get_json()
    assert set(body.keys()) == {"prediction", "confidence"}
    assert isinstance(body["prediction"], int)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_batch_returns_predictions():
    client = create_client()
    response = client.post("/predict/batch", json={"transactions": [SAMPLE_PAYLOAD]})

    assert response.status_code == 200
    body = response.get_json()
    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["transaction_id"] == SAMPLE_PAYLOAD["transaction_id"]


def test_model_info_returns_bundle_metadata():
    client = create_client()
    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.get_json()
    assert body["model_name"] == "mcc-transaction-classifier"
    assert "model_version" in body
    assert "metrics" in body


def test_predict_rejects_missing_required_field():
    client = create_client()
    invalid_payload = {key: value for key, value in SAMPLE_PAYLOAD.items() if key != "city"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing required field: city"}


def test_predict_rejects_invalid_amount_type():
    client = create_client()
    invalid_payload = {**SAMPLE_PAYLOAD, "amount": "450"}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "amount must be positive number"}


def test_predict_rejects_negative_amount():
    client = create_client()
    invalid_payload = {**SAMPLE_PAYLOAD, "amount": -10.0}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "amount must be positive number"}


def test_predict_rejects_empty_items():
    client = create_client()
    invalid_payload = {**SAMPLE_PAYLOAD, "items": []}

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "items must be a non-empty list"}


def test_predict_rejects_non_json_request():
    client = create_client()
    response = client.post("/predict", data="plain text", content_type="text/plain")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "request content type must be application/json",
    }


def test_predict_rejects_empty_json_body():
    client = create_client()
    response = client.post("/predict", data="", content_type="application/json")

    assert response.status_code == 400
    assert response.get_json() == {"error": "payload must be a JSON object"}


def test_predict_batch_rejects_bad_payload():
    client = create_client()
    response = client.post("/predict/batch", json={"transactions": "not-a-list"})

    assert response.status_code == 400
    assert response.get_json() == {"error": "transactions must be a non-empty list"}
