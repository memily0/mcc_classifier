from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def get_score_matrix(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            return np.column_stack([-scores, scores])
        return scores

    raise ValueError("model must expose predict_proba or decision_function")


def compute_multiclass_metrics(model, X, y_true) -> Dict[str, float]:
    predictions = model.predict(X)
    score_matrix = get_score_matrix(model, X)
    labels = model.named_steps["classifier"].classes_

    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "top3_accuracy": round(
            float(top_k_accuracy_score(y_true, score_matrix, k=3, labels=labels)),
            4,
        ),
        "macro_f1": round(float(f1_score(y_true, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, predictions, average="weighted")), 4),
        "mean_confidence": round(float(np.max(score_matrix, axis=1).mean()), 4),
    }


def summarize_top_confusions(y_true, y_pred, labels, top_n: int = 5) -> List[Dict[str, Any]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusions: List[Dict[str, Any]] = []

    for actual_idx, actual_label in enumerate(labels):
        for predicted_idx, predicted_label in enumerate(labels):
            if actual_idx == predicted_idx:
                continue
            count = int(matrix[actual_idx, predicted_idx])
            if count > 0:
                confusions.append({
                    "actual": int(actual_label),
                    "predicted": int(predicted_label),
                    "count": count,
                })

    return sorted(confusions, key=lambda row: row["count"], reverse=True)[:top_n]


def build_holdout_evaluation(model, X_test, y_test) -> Dict[str, Any]:
    predictions = model.predict(X_test)
    labels = model.named_steps["classifier"].classes_

    return {
        "metrics": compute_multiclass_metrics(model, X_test, y_test),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "top_confusions": summarize_top_confusions(y_test, predictions, labels),
    }
