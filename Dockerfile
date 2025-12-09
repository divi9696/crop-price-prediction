FROM python:3.11-slim

# -------------------- Environment --------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_MODE=api

WORKDIR /app

# -------------------- System deps & user --------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN groupadd -r appuser && useradd -r -g appuser appuser

# -------------------- Python deps --------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -------------------- App code --------------------
COPY src/ ./src/
COPY config.yml .
COPY app.py .
COPY run_api_direct.py .

# Data & models (ensure dirs exist in project; can be empty)
RUN mkdir -p /app/data /app/models /app/logs
COPY data/ ./data/
COPY models/ ./models/

# -------------------- Startup script --------------------
COPY start.sh /app/start.sh

# Fix possible Windows CRLF line endings
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# -------------------- Permissions & user --------------------
RUN chown -R appuser:appuser /app
USER appuser

# -------------------- Ports & entrypoint --------------------
EXPOSE 8001 8501
ENTRYPOINT ["/app/start.sh"]
