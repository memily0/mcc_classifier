# MCC Transaction Classifier

ML pet-project for MCC prediction from transaction metadata and basket items.

The API is implemented with Flask and serves a trained text+numeric classifier from `solution/model/model_bundle.pkl`.

## Project layout

- `solution/app.py` - Flask API with `/health`, `/predict`, `/predict/batch`
- `solution/features.py` - request validation and feature engineering
- `solution/train_model.py` - model training script
- `solution/model/` - saved model artifacts
- `tests/test_api.py` - smoke tests for API endpoints
- `Dockerfile` - root Docker build for local run

## Run locally with Docker

```bash
docker build -t mcc-classifier-local .
docker run --rm -p 8080:8080 mcc-classifier-local
```

Use `127.0.0.1` for local requests on macOS:

```bash
curl http://127.0.0.1:8080/health
```

## Example predict request

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-1001",
    "terminal_name": "SURF COFFEE",
    "terminal_description": "COFFEE SHOP",
    "city": "Moscow",
    "amount": 450.0,
    "items": [
      {"name": "latte", "price": 250.0},
      {"name": "croissant", "price": 200.0}
    ]
  }'
```

Example response:

```json
{
  "prediction": 5814,
  "confidence": 0.4721
}
```

## Train or refresh the model

```bash
cd solution
python3 train_model.py
```

The training script reports holdout metrics and updates `solution/model/model_bundle.pkl`.

## Run tests

```bash
pytest
```
