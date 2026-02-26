# Use lightweight Python image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (needed for some TF builds & Pillow)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend.py .

# Expose API port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_BUCKET=""
ENV MODEL_NAME=""

# Start FastAPI
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]