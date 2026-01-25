from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import prepare_data, select_numeric, select_char_text, select_full_text, BadRequestError

app = Flask(__name__)


model_cat = CatBoostClassifier()
model_cat.load_model("model/model_cat.cbm")

model_svc = joblib.load("model/model_svc.pkl")
model_lr_char = joblib.load("model/model_lr_char.pkl")
le = joblib.load("model/le_mcc.pkl")

W_SVC = 0.6
W_LR  = 0.25
W_CAT = 0.15


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def ensemble_predict(df: pd.DataFrame):
    numeric_features_ordered = [
        'items_total_price', 'items_mean_price', 'items_price_std',
        'items_min_price', 'items_max_price', 'items_price_range',
        'items_vs_amount', 'amount_log', 'terminal_name_len',
        'terminal_desc_len', 'items_text_len', 'amount_per_item',
        'items_price_skew'
    ]
   
    for col in numeric_features_ordered + ['full_text']:
        if col not in df.columns:
            df[col] = 0 if col != 'full_text' else ""
    
   
    X_cat = df[numeric_features_ordered + ['full_text']].copy()
    X_cat[numeric_features_ordered] = X_cat[numeric_features_ordered].astype(float)
    X_cat['full_text'] = X_cat['full_text'].astype(str)

    X_svc = df[numeric_features_ordered + ['char_text']].copy()
    X_svc[numeric_features_ordered] = X_svc[numeric_features_ordered].astype(float)
    X_svc['char_text'] = X_svc['char_text'].astype(str)

    X_lr = df[numeric_features_ordered + ['full_text']].copy()
    X_lr[numeric_features_ordered] = X_lr[numeric_features_ordered].astype(float)
    X_lr['full_text'] = X_lr['full_text'].astype(str)

    
    p_svc = model_svc.predict_proba(X_svc)
    p_lr  = model_lr_char.predict_proba(X_lr)
    p_cat = model_cat.predict_proba(X_cat) 


    p_ensemble = W_SVC * p_svc + W_LR * p_lr + W_CAT * p_cat

    idx = np.argmax(p_ensemble, axis=1)
    mcc_pred = le.inverse_transform(idx)
    confidence = np.max(p_ensemble, axis=1)

    return mcc_pred, confidence



@app.route("/predict", methods=["POST"])
def predict_single():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Empty request body"}), 400

    try:
        required_fields = ["transaction_id", "terminal_name", "terminal_description", "city", "amount", "items"]
        for field in required_fields:
            if field not in data:
                raise BadRequestError(f"Missing required field: {field}")

        items = data.get("items", [])
        if not isinstance(items, list):
            raise BadRequestError("items must be a list")

        if len(items) == 0:
            data["items"] = [{"name": "", "price": 0.01}]


        df = prepare_data({"transactions": [data]})

        mcc_pred, confidence = ensemble_predict(df)

        return jsonify({
            "transaction_id": data["transaction_id"],
            "predicted_mcc": int(mcc_pred[0]),
            "confidence": float(round(confidence[0], 3))
        })

    except BadRequestError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": "internal server error"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    predictions = []
    errors = []

    try:
        if "transactions" not in data:
            raise BadRequestError("missing 'transactions' key")
        if not isinstance(data["transactions"], list) or len(data["transactions"]) == 0:
            raise BadRequestError("'transactions' must be a non-empty list")

        for tx in data["transactions"]:
            tx_id = tx.get("transaction_id", None)
            if not tx_id:
                errors.append({"transaction_id": "unknown", "error": "missing transaction_id"})
                continue

            try:
                df = prepare_data({"transactions": [tx]})
                mcc_pred, confidence = ensemble_predict(df)

                predictions.append({
                    "transaction_id": tx_id,
                    "predicted_mcc": int(mcc_pred[0]),
                    "confidence": float(round(confidence[0], 3))
                })

            except BadRequestError as e:
                errors.append({"transaction_id": tx_id, "error": str(e)})
            except Exception as e:
                errors.append({"transaction_id": tx_id, "error": "internal server error"})

        return jsonify({
            "predictions": predictions,
            "errors": errors
        })

    except BadRequestError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "internal server error"}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": "mcc-ensemble-classifier",
        "model_version": "1.0.0",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
