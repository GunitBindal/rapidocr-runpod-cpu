# RapidOCR RunPod CPU Serverless - Ultra-Fast with OpenVINO + Pre-cached Models
FROM python:3.11-slim-bookworm

# Set environment variables for CPU optimization
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # OpenVINO optimizations
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    # RapidOCR will auto-download models to ~/.rapidocr
    HOME=/root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    rapidocr-onnxruntime==1.3.24 \
    runpod==1.8.1 \
    pillow==10.4.0 && \
    rm -rf /root/.cache/pip /tmp/*

WORKDIR /app

# Copy handler
COPY handler.py /app/handler.py
COPY prewarm.py /app/prewarm.py

# Pre-download RapidOCR models (PP-OCRv4: ~10-20MB total)
# This downloads models to ~/.rapidocr and pre-warms ONNX runtime
RUN python3 /app/prewarm.py && rm -rf /tmp/* /root/.cache/pip /app/prewarm.py

# Final cleanup
RUN rm -rf /tmp/* /root/.cache/* /var/tmp/*

CMD ["python3", "-u", "/app/handler.py"]
