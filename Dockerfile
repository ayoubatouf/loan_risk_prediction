# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --chown=appuser:appuser . .

RUN chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "src.serving.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL maintainer="ATOUF <atouf.ayoub.1@gmail.com>" \
      org.opencontainers.image.source="https://github.com/ayoubatouf" \
      org.opencontainers.image.description="Loan Risk Prediction"
