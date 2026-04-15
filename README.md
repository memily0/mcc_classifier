# MCC Transaction Classifier

End-to-end ML/backend showcase project for multiclass MCC prediction from transaction metadata and basket items.

Это pet-project / showcase-проект, в котором по данным транзакции и списку товаров предсказывается MCC-код. Репозиторий специально оформлен как полный pipeline: сравнение моделей, обучение, сохранение артефакта, запуск inference API, тесты и Docker.

The repository covers the full workflow:

- compare multiple linear models on the same text+numeric feature pipeline
- train the selected serving model
- save a reusable model bundle with metadata
- serve the bundle through a Flask API
- validate requests and test the project with `pytest`
- package and run the API in Docker

## Current Serving Model

The current serving model is **Calibrated LinearSVC** on top of a shared `TF-IDF + numeric features` pipeline.

Сейчас в API сервируется финально выбранная модель, а не просто первый рабочий baseline. Модель была выбрана после отдельного сравнения нескольких кандидатов на одинаковом feature pipeline.

Why this model was selected:

- it had the strongest cross-validation results on the primary metric `macro-F1`
- it also led on `accuracy`, `top-3 accuracy`, and `weighted-F1`
- calibration preserves `predict_proba`, so the existing API can still return `confidence`

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yaml
├── data/
│   └── data.csv
├── artifacts/
│   ├── model/
│   │   └── model_bundle.pkl
│   └── reports/
│       └── model_selection.json
├── scripts/
│   ├── compare_models.py
│   ├── evaluate_model.py
│   └── train_model.py
├── src/
│   └── mcc_classifier/
│       ├── api/
│       │   └── app.py
│       ├── config/
│       │   └── settings.py
│       ├── features/
│       │   └── feature_contract.py
│       ├── model_io/
│       │   └── bundle.py
│       └── training/
│           ├── compare_models.py
│           ├── evaluate.py
│           ├── evaluate_model.py
│           ├── model_registry.py
│           ├── pipeline.py
│           └── train.py
└── tests/
    ├── test_api.py
    ├── test_feature_contract.py
    └── test_model_selection.py
```

## ML Pipeline

Ниже описан реальный текущий pipeline проекта: от данных и признаков до обучения, сохранения модели и инференса. Это полезно как для ревью, так и для объяснения проекта на собеседовании.

### Data

- training data lives in `data/data.csv`
- target column: `true_mcc`
- the dataset contains raw text source columns and precomputed numeric feature columns

В проекте используется табличный датасет, где часть признаков уже лежит в готовом виде. Это ограничение не скрывается: train и inference contract описаны честно, без вида, что всё строится из полностью сырого источника одинаково.

### Features

Shared text feature:

- `terminal_name`
- `terminal_description`
- `terminal_city` for training / `city` for inference
- `items_text` for training / `items[].name` for inference

The text feature is normalized and combined into a single `text` column in one shared feature contract module.

Текстовая логика общая для train и inference, чтобы проект было проще защищать технически и чтобы не было скрытого рассинхрона между обучением и API.

Numeric features:

- `amount`
- `amount_log`
- `item_count`
- `items_total_price`
- `items_mean_price`
- `items_price_std`
- `items_min_price`
- `items_max_price`
- `items_price_range`
- `items_vs_amount`
- `terminal_name_len`
- `terminal_desc_len`
- `items_text_len`
- `amount_per_item`
- `items_price_skew`

Important limitation:

- training still relies on **precomputed numeric feature columns from the dataset**
- inference computes those numeric features from raw JSON payloads
- this is explicitly documented in the bundle metadata and feature contract rather than hidden

Полная parity между train и inference есть для text feature, но numeric часть в train всё ещё зависит от уже подготовленных колонок датасета.

### Training

- shared preprocessor:
  - `StandardScaler` for numeric features
  - `TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000)` for text
- the serving model is trained via `scripts/train_model.py`
- the saved bundle includes:
  - trained sklearn pipeline
  - version
  - selected model name
  - holdout metrics
  - feature contract metadata
  - evaluation summary
  - model selection summary

### Serving

API endpoints:

- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /model/info`

The API keeps the same request/response contract as earlier project stages.

## Model Selection

Этот раздел нужен, чтобы показать, что финальная модель выбрана через engineering process с кросс-валидацией и несколькими метриками.

Compared models:

- `LogisticRegression`
- `SGDClassifier(loss="log_loss")`
- `Calibrated LinearSVC`

Why this shortlist:

- all three are appropriate for sparse text + numeric multiclass classification
- all three fit naturally into a shared linear feature pipeline
- the calibrated SVC keeps serving compatibility by exposing `predict_proba`

Evaluation setup:

- stratified 5-fold cross-validation
- same shared feature pipeline for every candidate
- metrics:
  - accuracy
  - top-3 accuracy
  - macro-F1
  - weighted-F1

Primary metric:

- `macro-F1`

Tie-breakers:

- weighted-F1
- accuracy
- top-3 accuracy
- serving simplicity

### Cross-Validation Results

| Model | Mean Accuracy | Std Accuracy | Mean Top-3 Accuracy | Std Top-3 | Mean Macro-F1 | Std Macro-F1 | Mean Weighted-F1 | Std Weighted-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated LinearSVC | 0.9300 | 0.0149 | 0.9767 | 0.0110 | 0.9296 | 0.0146 | 0.9296 | 0.0146 |
| SGDClassifier (log_loss) | 0.8607 | 0.0199 | 0.9553 | 0.0169 | 0.8611 | 0.0202 | 0.8611 | 0.0202 |
| Logistic Regression | 0.8193 | 0.0295 | 0.9367 | 0.0155 | 0.8183 | 0.0297 | 0.8183 | 0.0297 |

Decision:

- **Calibrated LinearSVC** became the final serving model because it clearly won on the primary metric and also led on all secondary quality metrics.

## Local Setup

Install the project in editable mode with dev dependencies:

```bash
python3 -m pip install --no-build-isolation -e ".[dev]"
```

## Run Model Selection

```bash
python3 scripts/compare_models.py
```

This writes a selection report to `artifacts/reports/model_selection.json`.

## Train the Final Serving Model

```bash
python3 scripts/train_model.py
```

This retrains the selected serving model and updates `artifacts/model/model_bundle.pkl`.

## Evaluate the Saved Bundle

```bash
python3 scripts/evaluate_model.py
```

## Run Tests

```bash
python3 -m pytest
```

## Run the API Locally

```bash
gunicorn --bind 0.0.0.0:8080 mcc_classifier.api.app:app
```

Example request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-1001",
    "terminal_name": "SURF COFFEE",
    "terminal_description": "COFFEE SHOP",
    "city": "Moscow",
    "amount": 450.0,
    "items": [
      {"name": "latte", "price": 250.0},
      {"name": "croissant", "price": 200.0}
    ]
  }'
```

## Run with Docker

```bash
docker build -t mcc-classifier-local .
docker run --rm -p 8080:8080 mcc-classifier-local
```

Or with compose:

```bash
docker compose up --build
```
