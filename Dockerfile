FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY api ./api
COPY documents ./documents
COPY graph ./graph
COPY graph2 ./graph2
COPY llm_models ./llm_models
COPY tools ./tools
COPY utils ./utils

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
