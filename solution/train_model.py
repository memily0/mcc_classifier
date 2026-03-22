from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE, normalize_text

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "model" / "model_bundle.pkl"


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame[MODEL_TEXT_FEATURE] = (
        frame["terminal_name"].fillna("").map(normalize_text)
        + " "
        + frame["terminal_description"].fillna("").map(normalize_text)
        + " "
        + frame["terminal_city"].fillna("").map(normalize_text)
        + " "
        + frame["items_text"].fillna("").map(normalize_text)
    ).str.strip()
    return frame[MODEL_NUMERIC_FEATURES + [MODEL_TEXT_FEATURE]]


def train():
    df = pd.read_csv(DATA_PATH)
    X = build_training_frame(df)
    y = df["true_mcc"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), MODEL_NUMERIC_FEATURES),
            (
                "txt",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000),
                MODEL_TEXT_FEATURE,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, C=1.5)),
        ]
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    bundle = {
        "version": "2.0.0",
        "model": model,
        "metrics": {
            "holdout_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "holdout_top3_accuracy": round(
                float(
                    top_k_accuracy_score(
                        y_test,
                        probabilities,
                        k=3,
                        labels=model.named_steps["classifier"].classes_,
                    )
                ),
                4,
            ),
            "mean_confidence": round(float(probabilities.max(axis=1).mean()), 4),
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(bundle["metrics"])


if __name__ == "__main__":
    train()
