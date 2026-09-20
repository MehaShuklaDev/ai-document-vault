FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TIKTOKEN_CACHE_DIR=/opt/tiktoken
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && (python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" || echo "tiktoken prefetch skipped; runtime falls back to approx tokenizer")
COPY . .
RUN useradd -m vault && mkdir -p /data/blobs && chown -R vault:vault /app /data
USER vault
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
