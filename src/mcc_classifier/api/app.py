from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from mcc_classifier.config.settings import PROJECT_MODEL_NAME
from mcc_classifier.features.feature_contract import (
    BadRequestError,
    MODEL_INPUT_COLUMNS,
    prepare_data,
)
from mcc_classifier.model_io.bundle import load_model_bundle


def create_app() -> Flask:
    app = Flask(__name__)
    logging.basicConfig(level=logging.INFO)

    gunicorn_logger = logging.getLogger("gunicorn.error")
    if gunicorn_logger.handlers:
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
    else:
        app.logger.setLevel(logging.INFO)

    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    model_metrics = model_bundle.get("metrics", {})
    model_version = model_bundle.get("version", "unknown")

    def get_json_payload():
        if not request.is_json:
            raise BadRequestError("request content type must be application/json")

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequestError("payload must be a JSON object")

        return payload

    def predict_frame(df):
        model_input = df[MODEL_INPUT_COLUMNS].copy()
        probabilities = model.predict_proba(model_input)
        predictions = model.predict(model_input)
        confidence = probabilities.max(axis=1)
        return predictions, confidence

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict_single():
        app.logger.info("POST /predict")

        try:
            payload = get_json_payload()
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
            payload = get_json_payload()
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
            "model_name": PROJECT_MODEL_NAME,
            "model_version": model_version,
            "metrics": model_metrics,
        })

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
