# RapidOCR RunPod CPU Serverless - PP-OCRv5 Mobile with Pre-cached Models
FROM python:3.11-slim-bookworm

# Set environment variables for CPU optimization
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # CPU threading optimizations (set to match vCPU count)
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    # RapidOCR models cache location
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

# Copy files
COPY handler.py /app/handler.py
COPY prewarm.py /app/prewarm.py
COPY config_v5.yaml /app/config.yaml
COPY models/ /app/models/

# Pre-warm ONNX runtime with bundled PP-OCRv5 models (21MB total)
RUN python3 /app/prewarm.py && rm -rf /tmp/* /root/.cache/pip /app/prewarm.py

# Final cleanup
RUN rm -rf /tmp/* /root/.cache/* /var/tmp/*

CMD ["python3", "-u", "/app/handler.py"]
