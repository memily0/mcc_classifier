from pathlib import Path

import pytest

from mcc_classifier.config import settings
from mcc_classifier.model_io.bundle import load_model_bundle


def test_resolve_project_root_points_to_repo_root():
    assert settings.PROJECT_ROOT == Path.cwd().resolve()
    assert settings.MODEL_BUNDLE_PATH == Path.cwd().resolve() / "artifacts" / "model" / "model_bundle.pkl"


def test_load_model_bundle_raises_helpful_error_for_missing_path(tmp_path):
    missing_bundle = tmp_path / "artifacts" / "model" / "missing.pkl"

    with pytest.raises(FileNotFoundError, match="Train the model"):
        load_model_bundle(missing_bundle)
