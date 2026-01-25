FROM python:3.11-slim

WORKDIR /app

COPY solution/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY solution/app.py .
COPY solution/features.py .
COPY solution/model/ ./model/

EXPOSE 8080

CMD ["python", "app.py"]





   