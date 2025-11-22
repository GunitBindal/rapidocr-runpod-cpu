#!/usr/bin/env python3
"""Pre-download RapidOCR models and pre-warm ONNX runtime"""
import os
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

print('🚀 Pre-downloading RapidOCR models...', flush=True)
print(f'ENV: OMP_NUM_THREADS={os.getenv("OMP_NUM_THREADS", "not set")}', flush=True)
print(f'ENV: MKL_NUM_THREADS={os.getenv("MKL_NUM_THREADS", "not set")}', flush=True)

# Initialize RapidOCR (this triggers model download to ~/.rapidocr)
print('📥 Initializing RapidOCR engine...', flush=True)
engine = RapidOCR()

print('✅ Models downloaded to ~/.rapidocr', flush=True)

# Pre-warm with dummy inference to load models into memory
print('🔥 Pre-warming ONNX runtime...', flush=True)
dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

try:
    result = engine(np.array(dummy_img))
    print('✅ ONNX runtime pre-warmed!', flush=True)
    print(f'Models loaded: Detection, Recognition, Classification', flush=True)
except Exception as e:
    print(f'⚠️  Pre-warming skipped: {e}', flush=True)
    print('(Models will load on first request)', flush=True)

print('🎯 Optimization complete!', flush=True)
