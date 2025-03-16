# Use an official Python 3.10 base image
FROM python:3.10

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container (adjust if needed)
COPY . .

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    tensorboard \
    opencv-python \
    pycocotools  # For MS COCO dataset handling

# Run the evaluation script when the container starts
CMD ["python", "evaluate.py"]