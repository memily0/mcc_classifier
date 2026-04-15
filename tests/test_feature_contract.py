import pandas as pd
import pytest

from mcc_classifier.features.feature_contract import (
    BadRequestError,
    MODEL_NUMERIC_FEATURES,
    MODEL_TEXT_FEATURE,
    TRAIN_TARGET_COLUMN,
    build_feature_row,
    build_text_feature,
    build_training_frame,
    normalize_text,
    prepare_data,
    validate_training_dataframe,
)


def test_build_training_frame_uses_shared_text_logic():
    row = {
        "terminal_name": "STARBUCKS",
        "terminal_description": "COFFEE SHOP",
        "terminal_city": "Moscow",
        "items_text": "latte croissant",
        TRAIN_TARGET_COLUMN: 5814,
    }
    for feature_name in MODEL_NUMERIC_FEATURES:
        row[feature_name] = 1.0

    df = pd.DataFrame([row])
    frame = build_training_frame(df)

    assert list(frame.columns) == MODEL_NUMERIC_FEATURES + [MODEL_TEXT_FEATURE]
    assert frame.loc[0, MODEL_TEXT_FEATURE] == build_text_feature(
        "STARBUCKS",
        "COFFEE SHOP",
        "Moscow",
        "latte croissant",
    )


def test_build_text_feature_matches_inference_input_shape():
    tx = {
        "transaction_id": "tx-1001",
        "terminal_name": "STARBUCKS",
        "terminal_description": "COFFEE SHOP",
        "city": "Moscow",
        "amount": 450.0,
        "items": [
            {"name": "latte", "price": 250.0},
            {"name": "croissant", "price": 200.0},
        ],
    }

    row = build_feature_row(tx)
    expected_text = build_text_feature(
        tx["terminal_name"],
        tx["terminal_description"],
        tx["city"],
        " ".join(item["name"] for item in tx["items"]),
    )

    assert row[MODEL_TEXT_FEATURE] == expected_text


def test_prepare_data_returns_expected_columns():
    payload = {
        "transactions": [
            {
                "transaction_id": "tx-1001",
                "terminal_name": "STARBUCKS",
                "terminal_description": "COFFEE SHOP",
                "city": "Moscow",
                "amount": 450.0,
                "items": [
                    {"name": "latte", "price": 250.0},
                    {"name": "croissant", "price": 200.0},
                ],
            }
        ]
    }

    df = prepare_data(payload)

    assert list(df.columns) == ["transaction_id", *MODEL_NUMERIC_FEATURES, MODEL_TEXT_FEATURE]


def test_validate_training_dataframe_rejects_missing_columns():
    incomplete_df = pd.DataFrame(
        [
            {
                "terminal_name": "STARBUCKS",
                "terminal_description": "COFFEE SHOP",
                "terminal_city": "Moscow",
                TRAIN_TARGET_COLUMN: 5814,
            }
        ]
    )

    with pytest.raises(ValueError, match="training dataset is missing required columns"):
        validate_training_dataframe(incomplete_df)


def test_build_feature_row_rejects_bool_amount():
    payload = {
        "transaction_id": "tx-1001",
        "terminal_name": "STARBUCKS",
        "terminal_description": "COFFEE SHOP",
        "city": "Moscow",
        "amount": True,
        "items": [{"name": "latte", "price": 250.0}],
    }

    with pytest.raises(BadRequestError, match="amount must be positive number"):
        build_feature_row(payload)


def test_build_feature_row_normalizes_text_consistently():
    payload = {
        "transaction_id": "tx-2001",
        "terminal_name": "Surf Coffee!!!",
        "terminal_description": "Coffee-Shop",
        "city": "Moscow",
        "amount": 450.0,
        "items": [{"name": "Latte 2%", "price": 250.0}],
    }

    row = build_feature_row(payload)
    expected = normalize_text("Surf Coffee!!! Coffee-Shop Moscow Latte 2%")

    assert row[MODEL_TEXT_FEATURE] == expected
