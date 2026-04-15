from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from mcc_classifier.config.settings import CV_SPLITS, DATA_PATH, RANDOM_STATE, REPORTS_DIR
from mcc_classifier.features.feature_contract import TRAIN_TARGET_COLUMN, build_training_frame
from mcc_classifier.training.model_registry import CandidateModelSpec, get_candidate_model_specs
from mcc_classifier.training.pipeline import build_model_pipeline


PRIMARY_SELECTION_METRIC = "macro_f1"


def _top3_scorer(estimator, X, y) -> float:
    if hasattr(estimator, "predict_proba"):
        score_matrix = estimator.predict_proba(X)
    else:
        score_matrix = estimator.decision_function(X)
    labels = estimator.named_steps["classifier"].classes_
    return float(top_k_accuracy_score(y, score_matrix, k=3, labels=labels))


def _macro_f1_scorer(estimator, X, y) -> float:
    predictions = estimator.predict(X)
    return float(f1_score(y, predictions, average="macro"))


def _weighted_f1_scorer(estimator, X, y) -> float:
    predictions = estimator.predict(X)
    return float(f1_score(y, predictions, average="weighted"))


def build_model_selection_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(results)
    sort_columns = [
        "mean_macro_f1",
        "mean_weighted_f1",
        "mean_accuracy",
        "mean_top3_accuracy",
        "simplicity_rank",
    ]
    ascending = [False, False, False, False, True]
    return table.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def select_final_model(results_table: pd.DataFrame) -> Dict[str, Any]:
    best_row = results_table.iloc[0]
    return {
        "selected_model_name": best_row["model_key"],
        "selected_display_name": best_row["model"],
        "primary_metric": PRIMARY_SELECTION_METRIC,
        "selection_reason": (
            "Selected by highest mean macro-F1 across stratified cross-validation, "
            "with weighted-F1, accuracy, top-3 accuracy, and serving simplicity as tie-breakers."
        ),
    }


def compare_candidate_models(data_path: Path | None = None) -> pd.DataFrame:
    dataset_path = data_path or DATA_PATH
    df = pd.read_csv(dataset_path)
    X = build_training_frame(df)
    y = df[TRAIN_TARGET_COLUMN].astype(int)

    splitter = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    candidate_specs = get_candidate_model_specs()

    results: List[Dict[str, Any]] = []
    for spec in candidate_specs.values():
        pipeline = build_model_pipeline(spec.classifier_factory())
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splitter,
            scoring={
                "accuracy": "accuracy",
                "top3_accuracy": _top3_scorer,
                "macro_f1": _macro_f1_scorer,
                "weighted_f1": _weighted_f1_scorer,
            },
            n_jobs=None,
        )
        results.append({
            "model_key": spec.name,
            "model": spec.display_name,
            "mean_accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "std_accuracy": round(float(scores["test_accuracy"].std()), 4),
            "mean_top3_accuracy": round(float(scores["test_top3_accuracy"].mean()), 4),
            "std_top3_accuracy": round(float(scores["test_top3_accuracy"].std()), 4),
            "mean_macro_f1": round(float(scores["test_macro_f1"].mean()), 4),
            "std_macro_f1": round(float(scores["test_macro_f1"].std()), 4),
            "mean_weighted_f1": round(float(scores["test_weighted_f1"].mean()), 4),
            "std_weighted_f1": round(float(scores["test_weighted_f1"].std()), 4),
            "serving_notes": spec.serving_notes,
            "simplicity_rank": spec.simplicity_rank,
        })

    return build_model_selection_table(results)


def save_model_selection_report(results_table: pd.DataFrame, report_path: Path | None = None) -> Path:
    target_path = report_path or REPORTS_DIR / "model_selection.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary_metric": PRIMARY_SELECTION_METRIC,
        "results": results_table.to_dict(orient="records"),
        "selection": select_final_model(results_table),
    }
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_path


def render_results_table(results_table: pd.DataFrame) -> str:
    display_columns = [
        "model",
        "mean_accuracy",
        "std_accuracy",
        "mean_top3_accuracy",
        "std_top3_accuracy",
        "mean_macro_f1",
        "std_macro_f1",
        "mean_weighted_f1",
        "std_weighted_f1",
    ]
    return results_table[display_columns].to_string(index=False)


def main() -> None:
    results_table = compare_candidate_models()
    report_path = save_model_selection_report(results_table)
    selection = select_final_model(results_table)

    print(render_results_table(results_table))
    print()
    print(f"Primary metric: {PRIMARY_SELECTION_METRIC}")
    print(f"Selected model: {selection['selected_display_name']} ({selection['selected_model_name']})")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
