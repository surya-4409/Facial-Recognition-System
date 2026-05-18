FROM python:3.10-slim

LABEL maintainer="Facial Verification Fairness Audit"
LABEL description="Facial verification system with demographic bias auditing"

WORKDIR /app

# Install necessary system libraries for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .

# Stage 1: Install PyTorch CPU-only (much smaller ~170MB vs 670MB CUDA wheel)
# Using the official PyTorch CPU index for reliable, fast downloads
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Stage 2: Install remaining dependencies
RUN pip install --no-cache-dir --timeout 300 \
    facenet-pytorch==2.5.3 \
    opencv-python-headless==4.8.1.78 \
    Pillow==10.1.0 \
    numpy==1.26.2 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    gdown==4.7.3 \
    fpdf2==2.7.6 \
    tqdm==4.66.1 \
    requests==2.31.0

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts results submission

# Health check verifies Python environment and critical ML libraries
HEALTHCHECK --interval=30s --timeout=60s --start-period=120s --retries=5 \
    CMD python -c "import torch; import facenet_pytorch; import cv2; import pandas; print('Environment OK - torch:', torch.__version__)" || exit 1

# Default command runs the full pipeline
CMD ["python", "main.py"]
