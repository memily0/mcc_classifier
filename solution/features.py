from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

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
MAX_TEXT_LEN = 300


class BadRequestError(Exception):
    pass


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_LEN]


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
        if not isinstance(price, (int, float)) or price <= 0:
            raise BadRequestError("item price must be positive number")

        validated.append({
            "name": str(item.get("name", "")),
            "price": float(price),
        })

    return validated


def build_text(tx: Dict[str, Any], item_names: Iterable[str]) -> str:
    parts = [
        tx.get("terminal_name", ""),
        tx.get("terminal_description", ""),
        tx.get("city", ""),
        " ".join(item_names),
    ]
    return normalize_text(" ".join(str(part) for part in parts if part))


def build_feature_row(tx: Dict[str, Any]) -> Dict[str, Any]:
    tx = _require_transaction(tx)

    required_fields = [
        "transaction_id",
        "terminal_name",
        "terminal_description",
        "city",
        "amount",
        "items",
    ]
    for field in required_fields:
        if field not in tx:
            raise BadRequestError(f"Missing required field: {field}")

    amount = tx.get("amount")
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise BadRequestError("amount must be positive number")

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

    terminal_name = str(tx.get("terminal_name") or "")
    terminal_description = str(tx.get("terminal_description") or "")
    items_text = normalize_text(" ".join(item_names))
    text = build_text(tx, item_names)

    return {
        "transaction_id": str(tx["transaction_id"]),
        "amount": float(amount),
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

    expected_columns = ["transaction_id", *MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE]
    for column in expected_columns:
        if column not in df.columns:
            raise BadRequestError(f"failed to build feature column: {column}")

    return df
