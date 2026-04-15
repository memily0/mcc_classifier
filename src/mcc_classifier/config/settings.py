from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV_VAR = "MCC_CLASSIFIER_PROJECT_ROOT"


def _iter_candidate_roots() -> list[Path]:
    module_path = Path(__file__).resolve()
    cwd_path = Path.cwd().resolve()

    candidates: list[Path] = []

    env_root = os.getenv(PROJECT_ROOT_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    candidates.extend(module_path.parents)
    candidates.extend(cwd_path.parents)
    candidates.append(cwd_path)
    candidates.append(Path("/app"))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    return unique_candidates


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").exists()
        or (path / "src" / "mcc_classifier").exists()
        or (path / "artifacts").exists()
    )


def resolve_project_root() -> Path:
    for candidate in _iter_candidate_roots():
        if _looks_like_project_root(candidate):
            return candidate

    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "model" / "model_bundle.pkl"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"

MODEL_VERSION = "3.1.0"
PROJECT_MODEL_NAME = "mcc-transaction-classifier"
FINAL_SERVING_MODEL_NAME = "calibrated_linear_svc"

RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.2
CV_SPLITS = 5
