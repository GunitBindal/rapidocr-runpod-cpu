#!/usr/bin/env python3
"""
RapidOCR HTTP Server Handler for RunPod Load Balancer
Runs an HTTP server on PORT with /ping health check on PORT_HEALTH
"""
import os
import base64
import io
import json
import numpy as np
from PIL import Image
from rapidocr import RapidOCR
from flask import Flask, request, jsonify

print("Starting RapidOCR HTTP Server...", flush=True)
print(f"ENV: OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'not set')}", flush=True)
print(f"ENV: MKL_NUM_THREADS={os.getenv('MKL_NUM_THREADS', 'not set')}", flush=True)

# Get ports from environment
PORT = int(os.getenv('PORT', 8000))
PORT_HEALTH = int(os.getenv('PORT_HEALTH', 8001))

# Global engine instance
OCR_ENGINE = None

def initialize_engine():
    global OCR_ENGINE

    if OCR_ENGINE is None:
        print("Loading RapidOCR engine with PP-OCRv5...", flush=True)
        OCR_ENGINE = RapidOCR(params={
            "Det.model_path": "/app/models/ch_PP-OCRv5_mobile_det.onnx",
            "Rec.model_path": "/app/models/ch_PP-OCRv5_rec_mobile_infer.onnx",
            "Cls.model_path": "/app/models/ch_ppocr_mobile_v2.0_cls_infer.onnx"
        })
        print("✓ PP-OCRv5 engine loaded successfully!", flush=True)
        print("Models: PP-OCRv5 Mobile (21MB bundled)", flush=True)

    return OCR_ENGINE

# Initialize engine on startup
engine = initialize_engine()

# Create Flask app
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def health_check():
    """Health check endpoint for RunPod Load Balancer"""
    return jsonify({"status": "healthy", "engine": "RapidOCR PP-OCRv5"}), 200

@app.route('/', methods=['POST'])
def process_ocr():
    """Main OCR processing endpoint"""
    try:
        # Get input
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        images_b64 = data.get("images", [])

        if not images_b64:
            return jsonify({"success": False, "error": "No images provided"}), 400

        # Handle single image string
        if isinstance(images_b64, str):
            images_b64 = [images_b64]

        # Process images
        results = []
        for idx, img_b64 in enumerate(images_b64):
            try:
                # Remove data URL prefix if present
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",")[1]

                # Decode and convert to numpy array
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_array = np.array(img)
                print(f"✓ Image {idx+1} decoded: {img_array.shape}", flush=True)

                # Run OCR
                result = engine(img_array)

                # Format results
                text_lines = []
                if result and hasattr(result, 'dt_boxes') and result.dt_boxes is not None:
                    for box, text, score in zip(result.dt_boxes, result.txts, result.scores):
                        # Convert polygon to bbox
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)

                        text_lines.append({
                            "text": text,
                            "confidence": float(score),
                            "bbox": {
                                "x": int(x_min),
                                "y": int(y_min),
                                "width": int(x_max - x_min),
                                "height": int(y_max - y_min)
                            },
                            "polygon": [[int(p[0]), int(p[1])] for p in box]
                        })

                results.append({
                    "text_lines": text_lines,
                    "image_index": idx,
                    "total_lines": len(text_lines)
                })
                print(f"✓ Image {idx+1}: {len(text_lines)} text lines detected", flush=True)

            except Exception as e:
                print(f"✗ Image {idx+1} processing failed: {e}", flush=True)
                return jsonify({"success": False, "error": f"Image {idx+1} processing failed: {str(e)}"}), 500

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"✗ Handler error: {e}\n{error_trace}", flush=True)
        return jsonify({"success": False, "error": str(e), "traceback": error_trace}), 500

if __name__ == '__main__':
    print(f"Starting HTTP server on port {PORT}...", flush=True)
    print(f"Health check endpoint on port {PORT_HEALTH}...", flush=True)

    # Run health check server in separate thread
    from threading import Thread
    health_app = Flask('health')

    @health_app.route('/ping', methods=['GET'])
    def health():
        return jsonify({"status": "healthy"}), 200

    health_thread = Thread(target=lambda: health_app.run(host='0.0.0.0', port=PORT_HEALTH, debug=False))
    health_thread.daemon = True
    health_thread.start()

    print(f"✓ Health server running on port {PORT_HEALTH}", flush=True)

    # Run main server
    app.run(host='0.0.0.0', port=PORT, debug=False)
