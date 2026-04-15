from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mcc_classifier.features.feature_contract import MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
            (
                "txt",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000),
                MODEL_TEXT_FEATURE,
            ),
        ]
    )


def build_model_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
