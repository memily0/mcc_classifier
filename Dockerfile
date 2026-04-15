FROM python:3.11-slim

WORKDIR /app
ENV MCC_CLASSIFIER_PROJECT_ROOT=/app

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "--error-logfile", "-", "mcc_classifier.api.app:app"]
