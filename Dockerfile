FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app
COPY data ./data

# Cloud Run expects the server on port 8080
ENV PORT=8080

# Start FastAPI with Uvicorn
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

