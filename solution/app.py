from __future__ import annotations

import logging
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from features import BadRequestError, MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE, prepare_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = BASE_DIR / "model" / "model_bundle.pkl"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

gunicorn_logger = logging.getLogger("gunicorn.error")
if gunicorn_logger.handlers:
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
else:
    app.logger.setLevel(logging.INFO)

model_bundle = joblib.load(MODEL_BUNDLE_PATH)
model = model_bundle["model"]
model_metrics = model_bundle.get("metrics", {})
model_version = model_bundle.get("version", "2.0.0")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def predict_frame(df):
    model_input = df[MODEL_NUMERIC_FEATURES + [MODEL_TEXT_FEATURE]].copy()
    probabilities = model.predict_proba(model_input)
    predictions = model.predict(model_input)
    confidence = probabilities.max(axis=1)
    return predictions, confidence


@app.route("/predict", methods=["POST"])
def predict_single():
    app.logger.info("POST /predict")

    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequestError("payload must be a JSON object")

        app.logger.info("Building features for transaction_id=%s", payload.get("transaction_id"))
        df = prepare_data({"transactions": [payload]})

        app.logger.info(
            "Running model inference for transaction_id=%s",
            payload.get("transaction_id"),
        )
        predictions, confidence = predict_frame(df)

        return jsonify({
            "prediction": int(predictions[0]),
            "confidence": float(round(confidence[0], 4)),
        })

    except BadRequestError as exc:
        app.logger.warning("Validation error in /predict: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unhandled error in /predict")
        return jsonify({"error": "internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    app.logger.info("POST /predict/batch")

    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequestError("payload must be a JSON object")

        df = prepare_data(payload)
        predictions, confidence = predict_frame(df)

        results = []
        for idx, row in df.reset_index(drop=True).iterrows():
            results.append({
                "transaction_id": row["transaction_id"],
                "prediction": int(predictions[idx]),
                "confidence": float(round(confidence[idx], 4)),
            })

        return jsonify({"predictions": results})

    except BadRequestError as exc:
        app.logger.warning("Validation error in /predict/batch: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unhandled error in /predict/batch")
        return jsonify({"error": "internal server error"}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": "mcc-transaction-classifier",
        "model_version": model_version,
        "metrics": model_metrics,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
