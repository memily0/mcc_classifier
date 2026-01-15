from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    text = f"{data['terminal_description']} {' '.join([item['name'] for item in data['items']])}"
    text_features = vectorizer.transform([text]).toarray()
    
    amount = data["amount"]
    items = data["items"]
    avg_price = np.mean([item["price"] for item in items])
    item_count = len(items)
    
    features = np.hstack([text_features, np.array([[amount, avg_price, item_count]])])
    features = scaler.transform(features)
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))
    
    return jsonify({
        "transaction_id": data["transaction_id"],
        "predicted_mcc": int(prediction),
        "confidence": round(confidence, 3)
    })

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    transactions = data["transactions"]
    
    predictions = []
    for tx in transactions:
        text = f"{tx['terminal_description']} {' '.join([item['name'] for item in tx['items']])}"
        text_features = vectorizer.transform([text]).toarray()
        
        amount = tx["amount"]
        items = tx["items"]
        avg_price = np.mean([item["price"] for item in items])
        item_count = len(items)
        
        features = np.hstack([text_features, np.array([[amount, avg_price, item_count]])])
        features = scaler.transform(features)
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        predictions.append({
            "transaction_id": tx["transaction_id"],
            "predicted_mcc": int(prediction),
            "confidence": round(confidence, 3)
        })
    
    return jsonify({"predictions": predictions})

@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name": "mcc-classifier",
        "model_version": "1.0.0"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
