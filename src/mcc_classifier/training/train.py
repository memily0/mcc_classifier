from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from mcc_classifier import __version__
from mcc_classifier.config.settings import (
    DATA_PATH,
    HOLDOUT_TEST_SIZE,
    MODEL_VERSION,
    RANDOM_STATE,
)
from mcc_classifier.features.feature_contract import (
    TRAIN_TARGET_COLUMN,
    build_feature_contract_metadata,
    build_training_frame,
)
from mcc_classifier.model_io.bundle import save_model_bundle
from mcc_classifier.training.compare_models import PRIMARY_SELECTION_METRIC
from mcc_classifier.training.evaluate import build_holdout_evaluation
from mcc_classifier.training.model_registry import get_serving_model_spec
from mcc_classifier.training.pipeline import build_model_pipeline


def train_and_save_serving_model() -> dict:
    df = pd.read_csv(DATA_PATH)
    X = build_training_frame(df)
    y = df[TRAIN_TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    serving_spec = get_serving_model_spec()
    model = build_model_pipeline(serving_spec.classifier_factory())
    model.fit(X_train, y_train)

    evaluation = build_holdout_evaluation(model, X_test, y_test)

    bundle = {
        "version": MODEL_VERSION,
        "package_version": __version__,
        "model_name": serving_spec.name,
        "display_name": serving_spec.display_name,
        "model": model,
        "metrics": {
            "holdout_accuracy": evaluation["metrics"]["accuracy"],
            "holdout_top3_accuracy": evaluation["metrics"]["top3_accuracy"],
            "holdout_macro_f1": evaluation["metrics"]["macro_f1"],
            "holdout_weighted_f1": evaluation["metrics"]["weighted_f1"],
            "mean_confidence": evaluation["metrics"]["mean_confidence"],
        },
        "evaluation": evaluation,
        "feature_contract": build_feature_contract_metadata(),
        "selection_summary": {
            "primary_metric": PRIMARY_SELECTION_METRIC,
            "selected_model_name": serving_spec.name,
            "selected_model_display_name": serving_spec.display_name,
            "serving_notes": serving_spec.serving_notes,
        },
        "training_summary": {
            "dataset_rows": int(len(df)),
            "num_classes": int(y.nunique()),
            "test_size": HOLDOUT_TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
    }

    save_model_bundle(bundle)
    return bundle


def main() -> None:
    bundle = train_and_save_serving_model()
    print(bundle["metrics"])


if __name__ == "__main__":
    main()
