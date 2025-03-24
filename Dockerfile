# Use an official Python 3.10 base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/best_model.pth \
    BATCH_SIZE=32 \
    NUM_WORKERS=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories for mounting
RUN mkdir -p /app/data/images /app/data/annotations /app/models

# Healthcheck to verify the container is running properly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command to run evaluation
ENTRYPOINT ["python", "evaluate.py"]
CMD ["--model_path", "${MODEL_PATH}", \
     "--batch_size", "${BATCH_SIZE}", \
     "--num_workers", "${NUM_WORKERS}"]