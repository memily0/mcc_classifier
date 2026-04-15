from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib

from mcc_classifier.config.settings import MODEL_BUNDLE_PATH


def load_model_bundle(path: Path | None = None) -> Dict[str, Any]:
    bundle_path = path or MODEL_BUNDLE_PATH
    if not bundle_path.exists():
        raise FileNotFoundError(
            "Model bundle not found at "
            f"'{bundle_path}'. Train the model with `python3 scripts/train_model.py` "
            "or ensure `artifacts/model/model_bundle.pkl` is present in the runtime environment."
        )
    return joblib.load(bundle_path)


def save_model_bundle(bundle: Dict[str, Any], path: Path | None = None) -> Path:
    bundle_path = path or MODEL_BUNDLE_PATH
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle_path
