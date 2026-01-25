import pandas as pd
import re
from typing import List, Dict, Any
import numpy as np

numeric_features_ordered = [
        'items_total_price', 'items_mean_price', 'items_price_std',
        'items_min_price', 'items_max_price', 'items_price_range',
        'items_vs_amount', 'amount_log', 'terminal_name_len',
        'terminal_desc_len', 'items_text_len', 'amount_per_item',
        'items_price_skew'
    ]

max_text_len = 150

class BadRequestError(Exception):
    pass

def select_char_text(X):
    return X['char_text']

def select_full_text(X):
    return X['full_text']

def select_numeric(X):
    return X[numeric_features_ordered]

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_text_len]

def safe_text(value: Any, max_len: int = 10_0) -> str:

    if value is None:
        return ""
    
    text = str(value)
    return text[:max_len]

def build_chars_text(tx: Dict[str, Any]) -> str:
    parts: List[str] = []

    if tx.get("terminal_name"):
        parts.append(tx["terminal_name"])

    if tx.get("terminal_description"):
        parts.append(tx["terminal_description"])

    items = tx.get("items", [])
    if isinstance(items, list):
        for item in items:
            name = item.get("name")
            if isinstance(name, str):
                parts.append(name)

    raw_text = " ".join(parts)
    return normalize_text(raw_text)

def build_full_text(tx: Dict[str, Any]) -> str:
    parts: List[str] = []

    if tx.get("terminal_name"):
        parts.append(tx["terminal_name"])

    if tx.get("terminal_description"):
        parts.append(tx["terminal_description"])

    if tx.get("city"):
        parts.append(tx["city"])

    items = tx.get("items", [])
    if isinstance(items, list):
        for item in items:
            name = item.get("name")
            if isinstance(name, str):
                parts.append(name)

    raw_text = " ".join(parts)
    return normalize_text(raw_text)


def build_numeric_features(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for tx in transactions:
        items = tx.get("items")                 

        if not isinstance(items, list) or len(items) == 0:
            raise BadRequestError("items must be a non-empty list")

        prices = []
        item_texts = []

        for item in items:
            if "price" not in item or "name" not in item:
                raise BadRequestError("each item must contain name and price")

            price = item["price"]

            if not isinstance(price, (int, float)) or price <= 0:
                raise BadRequestError("item price must be positive number")

            prices.append(price)
            item_texts.append(str(item["name"]))

        prices = np.array(prices, dtype=float)

        amount = tx.get("amount")
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise BadRequestError("amount must be positive number")

        item_count = len(prices)

        items_total_price = prices.sum()
        items_mean_price = prices.mean()
        items_min_price = prices.min()
        items_max_price = prices.max()
        items_price_std = prices.std(ddof=0)
        items_price_range = items_max_price - items_min_price

        amount_per_item = amount / (item_count + 1e-3)
        items_vs_amount = items_total_price / amount
        amount_log = np.log1p(amount)
        items_price_skew = (
            (items_max_price - items_mean_price)
            / (items_mean_price + 1e-3)
        )

        terminal_name = tx.get("terminal_name") or ""
        terminal_desc = tx.get("terminal_description") or ""

        rows.append({
            "items_total_price": items_total_price,
            "items_mean_price": items_mean_price,
            "items_price_std": items_price_std,
            "items_min_price": items_min_price,
            "items_max_price": items_max_price,
            "items_price_range": items_price_range,
            "items_vs_amount": items_vs_amount,
            "amount_log": amount_log,
            "terminal_name_len": len(str(terminal_name)),
            "terminal_desc_len": len(str(terminal_desc)),
            "items_text_len": len(" ".join(item_texts)),
            "amount_per_item": amount_per_item,
            "items_price_skew": items_price_skew,
        })

    return pd.DataFrame(rows)


def prepare_data(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise BadRequestError("payload must be a JSON object")

    transactions = payload.get("transactions")

    if not isinstance(transactions, list) or len(transactions) == 0:
        raise BadRequestError("transactions must be a non-empty list")

    char_texts = []
    full_texts = []

    for tx in transactions:
        if not isinstance(tx, dict):
            raise BadRequestError("each transaction must be an object")

        char_texts.append(build_chars_text(tx))
        full_texts.append(build_full_text(tx))

    numeric_df = build_numeric_features(transactions)

    df = numeric_df.copy()
    df["char_text"] = char_texts
    df["full_text"] = full_texts

    return df


        


