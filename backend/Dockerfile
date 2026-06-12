# Backend image. Build from the repo root so the `backend` package resolves:
#   docker build -f backend/Dockerfile -t dispatch-iq-api .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first for layer caching.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Application code + the ML/pipeline packages it will serve from.
COPY backend ./backend
COPY pipeline ./pipeline
COPY ml ./ml
COPY mlops ./mlops

EXPOSE 8000

# Lightweight container healthcheck against the API root.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
