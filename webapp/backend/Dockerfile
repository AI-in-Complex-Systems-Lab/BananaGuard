FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV MODEL_PATH=/models/weapon_detection.pt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY webapp/backend/*.py .
COPY webapp/backend/models/weapon_detection.pt /models/weapon_detection.pt

EXPOSE 8080

CMD ["python", "server.py"]