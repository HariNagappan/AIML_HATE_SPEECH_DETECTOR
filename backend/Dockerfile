# CPU-friendly image for the research backend.
# Build:  docker build -t hate-speech-backend .
# Run:    docker run -p 8000:8000 hate-speech-backend
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv/backend

# Install dependencies first (better docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source.
COPY . .

EXPOSE 8000

# Load a trained checkpoint by mounting it and setting MODEL_CHECKPOINT
# (without one, the API starts but reports untrained heads).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
