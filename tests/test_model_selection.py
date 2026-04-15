import pandas as pd

from mcc_classifier.training.compare_models import build_model_selection_table, select_final_model


def test_model_selection_table_sorts_by_primary_metric_then_tie_breakers():
    results = [
        {
            "model_key": "sgd_log_loss",
            "model": "SGDClassifier (log_loss)",
            "mean_accuracy": 0.82,
            "std_accuracy": 0.01,
            "mean_top3_accuracy": 0.93,
            "std_top3_accuracy": 0.01,
            "mean_macro_f1": 0.8,
            "std_macro_f1": 0.02,
            "mean_weighted_f1": 0.81,
            "std_weighted_f1": 0.01,
            "serving_notes": "fast",
            "simplicity_rank": 2,
        },
        {
            "model_key": "logistic_regression",
            "model": "Logistic Regression",
            "mean_accuracy": 0.83,
            "std_accuracy": 0.02,
            "mean_top3_accuracy": 0.94,
            "std_top3_accuracy": 0.01,
            "mean_macro_f1": 0.82,
            "std_macro_f1": 0.01,
            "mean_weighted_f1": 0.82,
            "std_weighted_f1": 0.01,
            "serving_notes": "simple",
            "simplicity_rank": 1,
        },
    ]

    table = build_model_selection_table(results)

    assert isinstance(table, pd.DataFrame)
    assert list(table["model_key"]) == ["logistic_regression", "sgd_log_loss"]


def test_select_final_model_returns_top_ranked_candidate():
    table = pd.DataFrame(
        [
            {
                "model_key": "logistic_regression",
                "model": "Logistic Regression",
                "mean_accuracy": 0.83,
                "std_accuracy": 0.02,
                "mean_top3_accuracy": 0.94,
                "std_top3_accuracy": 0.01,
                "mean_macro_f1": 0.82,
                "std_macro_f1": 0.01,
                "mean_weighted_f1": 0.82,
                "std_weighted_f1": 0.01,
                "serving_notes": "simple",
                "simplicity_rank": 1,
            }
        ]
    )

    selection = select_final_model(table)

    assert selection["selected_model_name"] == "logistic_regression"
    assert selection["primary_metric"] == "macro_f1"
