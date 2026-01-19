# Stage 1: Build dependencies and train model
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter AS builder

WORKDIR /app

# Copy only requirements first to leverage caching
COPY ./app/requirements.txt .

# Install dependencies in a cached layer
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script and data
COPY ./app/train.py .
COPY ./data ./data

# Train all pneumonia model
RUN python train.py

# Stage 2: Runtime environment (smaller image)
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies only (minimal footprint)
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files from the build stage
COPY ./app/backend.py .

# Copy trained model
COPY --from=builder /app/pneumonia_model.h5 .

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
