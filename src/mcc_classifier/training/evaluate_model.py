from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from mcc_classifier.config.settings import DATA_PATH, HOLDOUT_TEST_SIZE, RANDOM_STATE
from mcc_classifier.features.feature_contract import TRAIN_TARGET_COLUMN, build_training_frame
from mcc_classifier.model_io.bundle import load_model_bundle
from mcc_classifier.training.evaluate import build_holdout_evaluation


def evaluate_saved_model() -> dict:
    bundle = load_model_bundle()
    model = bundle["model"]

    df = pd.read_csv(DATA_PATH)
    X = build_training_frame(df)
    y = df[TRAIN_TARGET_COLUMN].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return build_holdout_evaluation(model, X_test, y_test)


def main() -> None:
    print(json.dumps(evaluate_saved_model(), indent=2))


if __name__ == "__main__":
    main()
