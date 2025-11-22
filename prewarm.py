#!/usr/bin/env python3
"""Pre-warm ONNX runtime with bundled PP-OCRv5 models"""
import os
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

print('🚀 Pre-warming PP-OCRv5 models...', flush=True)
print(f'ENV: OMP_NUM_THREADS={os.getenv("OMP_NUM_THREADS", "not set")}', flush=True)
print(f'ENV: MKL_NUM_THREADS={os.getenv("MKL_NUM_THREADS", "not set")}', flush=True)

# Initialize RapidOCR with bundled PP-OCRv5 models
print('📥 Loading bundled PP-OCRv5 models...', flush=True)
engine = RapidOCR(params={
    "Det.model_path": "/app/models/ch_PP-OCRv5_mobile_det.onnx",
    "Rec.model_path": "/app/models/ch_PP-OCRv5_rec_mobile_infer.onnx",
    "Cls.model_path": "/app/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
})
print('✅ PP-OCRv5 models loaded (21MB total)', flush=True)

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
