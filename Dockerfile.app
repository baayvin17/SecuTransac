FROM python:3.11-slim

WORKDIR /app

# Dépendances système utiles (xgboost peut en avoir besoin selon les wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Le code est monté via volume (./ -> /app)
