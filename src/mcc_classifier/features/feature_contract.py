from __future__ import annotations

import re
from math import isfinite
from typing import Any, Dict, List

import numpy as np
import pandas as pd

MODEL_NUMERIC_FEATURES = [
    "amount",
    "amount_log",
    "item_count",
    "items_total_price",
    "items_mean_price",
    "items_price_std",
    "items_min_price",
    "items_max_price",
    "items_price_range",
    "items_vs_amount",
    "terminal_name_len",
    "terminal_desc_len",
    "items_text_len",
    "amount_per_item",
    "items_price_skew",
]
MODEL_TEXT_FEATURE = "text"
MODEL_INPUT_COLUMNS = [*MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE]

MAX_TEXT_LEN = 300
TRAIN_TARGET_COLUMN = "true_mcc"
TRAIN_TEXT_SOURCE_COLUMNS = [
    "terminal_name",
    "terminal_description",
    "terminal_city",
    "items_text",
]
INFERENCE_REQUIRED_FIELDS = [
    "transaction_id",
    "terminal_name",
    "terminal_description",
    "city",
    "amount",
    "items",
]


class BadRequestError(Exception):
    """Raised when the inference payload does not satisfy the contract."""


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_LEN]


def build_text_feature(
    terminal_name: Any,
    terminal_description: Any,
    city: Any,
    items_text: Any,
) -> str:
    parts = [terminal_name, terminal_description, city, items_text]
    return normalize_text(" ".join(str(part) for part in parts if part))


def build_feature_contract_metadata() -> Dict[str, Any]:
    return {
        "shared_text_feature": {
            "name": MODEL_TEXT_FEATURE,
            "train_columns": TRAIN_TEXT_SOURCE_COLUMNS,
            "inference_fields": [
                "terminal_name",
                "terminal_description",
                "city",
                "items[].name",
            ],
        },
        "numeric_features": {
            "names": MODEL_NUMERIC_FEATURES,
            "train_source": "precomputed dataset columns",
            "inference_source": "computed from transaction amount and items[]",
        },
    }


def _is_valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and isfinite(float(value))
    )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise BadRequestError(f"{field_name} must be a non-empty string")

    normalized = value.strip()
    if not normalized:
        raise BadRequestError(f"{field_name} must be a non-empty string")

    return normalized


def validate_training_dataframe(df: pd.DataFrame) -> None:
    required_columns = [
        *TRAIN_TEXT_SOURCE_COLUMNS,
        *MODEL_NUMERIC_FEATURES,
        TRAIN_TARGET_COLUMN,
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"training dataset is missing required columns: {missing}")


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature frame used by training.

    Training expects a tabular dataset with:
    - raw text source columns used to build the shared `text` feature
    - precomputed numeric feature columns matching MODEL_NUMERIC_FEATURES

    This keeps the shared text feature logic identical to inference while being
    explicit that numeric features in training come from precomputed dataset
    columns rather than raw item lists.
    """

    validate_training_dataframe(df)

    frame = df.copy()
    frame[MODEL_TEXT_FEATURE] = frame.apply(
        lambda row: build_text_feature(
            row["terminal_name"],
            row["terminal_description"],
            row["terminal_city"],
            row["items_text"],
        ),
        axis=1,
    )
    return frame[MODEL_INPUT_COLUMNS]


def _require_transaction(tx: Any) -> Dict[str, Any]:
    if not isinstance(tx, dict):
        raise BadRequestError("each transaction must be an object")
    return tx


def _validate_items(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list) or len(items) == 0:
        raise BadRequestError("items must be a non-empty list")

    validated: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise BadRequestError("each item must be an object")
        if "name" not in item or "price" not in item:
            raise BadRequestError("each item must contain name and price")

        price = item["price"]
        if not _is_valid_number(price) or float(price) <= 0:
            raise BadRequestError("item price must be positive number")

        validated.append({
            "name": _require_non_empty_string(item.get("name"), "item name"),
            "price": float(price),
        })

    return validated


def build_feature_row(tx: Dict[str, Any]) -> Dict[str, Any]:
    tx = _require_transaction(tx)

    for field in INFERENCE_REQUIRED_FIELDS:
        if field not in tx:
            raise BadRequestError(f"Missing required field: {field}")

    transaction_id = _require_non_empty_string(tx.get("transaction_id"), "transaction_id")
    terminal_name = _require_non_empty_string(tx.get("terminal_name"), "terminal_name")
    terminal_description = _require_non_empty_string(
        tx.get("terminal_description"),
        "terminal_description",
    )
    city = _require_non_empty_string(tx.get("city"), "city")

    amount = tx.get("amount")
    if not _is_valid_number(amount) or float(amount) <= 0:
        raise BadRequestError("amount must be positive number")
    amount = float(amount)

    items = _validate_items(tx.get("items"))
    prices = np.array([item["price"] for item in items], dtype=float)
    item_names = [item["name"] for item in items]

    item_count = int(len(prices))
    items_total_price = float(prices.sum())
    items_mean_price = float(prices.mean())
    items_min_price = float(prices.min())
    items_max_price = float(prices.max())
    items_price_std = float(prices.std(ddof=0))
    items_price_range = float(items_max_price - items_min_price)
    items_vs_amount = float(items_total_price / amount)
    amount_log = float(np.log1p(amount))
    amount_per_item = float(amount / max(item_count, 1))
    items_price_skew = float(
        (items_max_price - items_mean_price) / (items_mean_price + 1e-3)
    )

    items_text = normalize_text(" ".join(item_names))
    text = build_text_feature(
        terminal_name,
        terminal_description,
        city,
        " ".join(item_names),
    )

    return {
        "transaction_id": transaction_id,
        "amount": amount,
        "amount_log": amount_log,
        "item_count": item_count,
        "items_total_price": items_total_price,
        "items_mean_price": items_mean_price,
        "items_price_std": items_price_std,
        "items_min_price": items_min_price,
        "items_max_price": items_max_price,
        "items_price_range": items_price_range,
        "items_vs_amount": items_vs_amount,
        "terminal_name_len": len(terminal_name),
        "terminal_desc_len": len(terminal_description),
        "items_text_len": len(items_text),
        "amount_per_item": amount_per_item,
        "items_price_skew": items_price_skew,
        "text": text,
    }


def prepare_data(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise BadRequestError("payload must be a JSON object")

    transactions = payload.get("transactions")
    if not isinstance(transactions, list) or len(transactions) == 0:
        raise BadRequestError("transactions must be a non-empty list")

    rows = [build_feature_row(tx) for tx in transactions]
    df = pd.DataFrame(rows)

    expected_columns = ["transaction_id", *MODEL_INPUT_COLUMNS]
    for column in expected_columns:
        if column not in df.columns:
            raise BadRequestError(f"failed to build feature column: {column}")

    return df
