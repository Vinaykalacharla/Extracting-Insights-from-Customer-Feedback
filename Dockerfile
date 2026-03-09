FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY interninfos/requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY interninfos/ .

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 10000

CMD ["sh", "-c", "gunicorn --workers ${WEB_CONCURRENCY:-1} --timeout ${GUNICORN_TIMEOUT:-120} --bind 0.0.0.0:${PORT:-10000} app:app"]
