from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from mcc_classifier.config.settings import FINAL_SERVING_MODEL_NAME, RANDOM_STATE


@dataclass(frozen=True)
class CandidateModelSpec:
    name: str
    display_name: str
    classifier_factory: Callable[[], object]
    serving_notes: str
    simplicity_rank: int


def get_candidate_model_specs() -> Dict[str, CandidateModelSpec]:
    return {
        "logistic_regression": CandidateModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            classifier_factory=lambda: LogisticRegression(
                max_iter=4000,
                C=1.5,
                random_state=RANDOM_STATE,
            ),
            serving_notes="Simple linear baseline with native predict_proba.",
            simplicity_rank=1,
        ),
        "sgd_log_loss": CandidateModelSpec(
            name="sgd_log_loss",
            display_name="SGDClassifier (log_loss)",
            classifier_factory=lambda: SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=3000,
                tol=1e-3,
                random_state=RANDOM_STATE,
            ),
            serving_notes="Fast linear model with probabilistic output, but usually less stable.",
            simplicity_rank=2,
        ),
        "calibrated_linear_svc": CandidateModelSpec(
            name="calibrated_linear_svc",
            display_name="Calibrated LinearSVC",
            classifier_factory=lambda: CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    dual="auto",
                    max_iter=10000,
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
            ),
            serving_notes="Strong sparse-text classifier with calibrated probabilities, but more complex serving/training.",
            simplicity_rank=3,
        ),
    }


def get_serving_model_spec() -> CandidateModelSpec:
    return get_candidate_model_specs()[FINAL_SERVING_MODEL_NAME]
