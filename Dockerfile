FROM python:3.11-slim

WORKDIR /app

COPY solution/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY solution/app.py ./app.py
COPY solution/features.py ./features.py
COPY solution/model/ ./model/

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
