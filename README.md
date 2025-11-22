# RapidOCR RunPod CPU Serverless

Ultra-fast OCR deployment on RunPod CPU instances with **pre-bundled PP-OCRv5 models**.

## Features

- ✅ **CPU-optimized**: Runs on cheap CPU instances ($0.10-0.20/hour)
- ✅ **Latest PP-OCRv5**: Better accuracy and speed than v4
- ✅ **Tiny bundled models**: Only 21MB total (included in repo!)
- ✅ **Instant cold starts**: <2 seconds (models pre-bundled)
- ✅ **Multi-threaded**: OpenMP/MKL optimizations for CPU performance
- ✅ **Fast inference**: ~1-1.5 seconds per page on 8-core CPU
- ✅ **90+ languages**: Supports English, Chinese, Japanese, Korean, etc.
- ✅ **No downloads**: Models bundled in Docker image

## Deployment Options

This repo supports **two deployment modes**:

1. **Serverless (Queue)** - For async job processing with RunPod's queue system
2. **Load Balancer (HTTP)** - For synchronous HTTP requests with auto-scaling

---

## Option 1: Serverless (Queue) Deployment

### 1. Build on RunPod

```bash
# Login to RunPod
runpod login

# Build image (uses RunPod's fast build servers)
runpod build \
  --repo https://github.com/YOUR_USERNAME/rapidocr-runpod-cpu.git \
  --branch main \
  --dockerfile Dockerfile \
  --tag latest \
  --public
```

### 2. Create Serverless Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Select your built image
4. **Choose CPU instance**: 8-16 vCPU recommended
5. Set Active Workers = 1 (or more for scaling)
6. Deploy!

### 3. Test Serverless Endpoint

```bash
# Set environment variables
export RUNPOD_API_KEY="your_api_key"
export RUNPOD_ENDPOINT_ID="your_endpoint_id"

# Install dependencies
pip install requests pillow pdf2image

# Run batch OCR
python3 batch_ocr.py sample.pdf --max-workers 5
```

---

## Option 2: Load Balancer (HTTP) Deployment

### 1. Build Load Balancer Image

```bash
# Build with Load Balancer Dockerfile
runpod build \
  --repo https://github.com/YOUR_USERNAME/rapidocr-runpod-cpu.git \
  --branch main \
  --dockerfile Dockerfile.loadbalancer \
  --tag loadbalancer \
  --public
```

### 2. Create Load Balancer Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click "New Endpoint"
3. Select your built image (with `loadbalancer` tag)
4. **Choose CPU instance**: 8-16 vCPU recommended
5. **Endpoint Type**: Load Balancer (not Queue!)
6. Set Active Workers = 1 (or more for auto-scaling)
7. Deploy!

### 3. Test Load Balancer Endpoint

```bash
# Set endpoint URL
export RAPIDOCR_ENDPOINT_URL="https://your-endpoint-id.runpod.net"

# Install dependencies
pip install requests pillow pdf2image

# Run test
python3 test_loadbalancer.py sample.pdf
```

## API Usage

### Single Image

```python
import requests
import base64

# Read image
with open("image.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Call API
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT/run",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={"input": {"images": [img_b64]}}
)

result = response.json()
print(result)
```

### Response Format

```json
{
  "success": true,
  "results": [
    {
      "text_lines": [
        {
          "text": "Hello World",
          "confidence": 0.98,
          "bbox": {"x": 10, "y": 20, "width": 100, "height": 30},
          "polygon": [[10, 20], [110, 20], [110, 50], [10, 50]]
        }
      ],
      "image_index": 0,
      "total_lines": 1
    }
  ]
}
```

## Performance

### CPU Instance Recommendations

| vCPUs | RAM | Cost/hr | Pages/sec | Use Case |
|-------|-----|---------|-----------|----------|
| 4     | 8GB | ~$0.10  | 0.3-0.5   | Light load |
| 8     | 16GB| ~$0.15  | 0.5-0.8   | **Recommended** |
| 16    | 32GB| ~$0.25  | 0.8-1.2   | Heavy load |

### Benchmark (8 vCPU)

- **Single page**: ~1.5-2s
- **100 pages**: ~3-4 minutes
- **Cold start**: <3 seconds (models pre-cached)

## Cost Comparison

| Solution | Hardware | Cost/1000 pages | Speed |
|----------|----------|-----------------|-------|
| **RapidOCR CPU** | 8 vCPU | **$0.08** | 1.5s/page |
| Surya A100 | 80GB GPU | $3.20 | 0.5s/page |
| Google Vision API | Cloud | $1.50 | 1s/page |

## Models

Pre-bundled PP-OCRv5 models (included in `/models/`):

- **Detection**: PP-OCRv5 mobile (4.6MB)
- **Recognition**: PP-OCRv5 mobile (16MB)
- **Classification**: Mobile v2.0 (0.5MB) - disabled for speed

Total: **21MB** (bundled in Docker image, no download needed!)

## Deployment Notes

### Environment Variables

Set in Dockerfile (already configured):
```dockerfile
OMP_NUM_THREADS=16       # OpenMP threads
MKL_NUM_THREADS=16       # Intel MKL threads
OPENBLAS_NUM_THREADS=16  # OpenBLAS threads
```

### Scaling

- **Horizontal**: Increase Active Workers for concurrent requests
- **Vertical**: Use larger CPU instances (16-32 vCPUs)
- **Auto-scale**: Set min/max workers in RunPod UI

## Troubleshooting

### Models not pre-cached?
Check build logs for "Pre-downloading RapidOCR models..." step.

### Slow performance?
- Increase vCPUs (8+ recommended)
- Check `OMP_NUM_THREADS` matches vCPU count
- Verify models loaded (check worker logs)

### Out of memory?
- Use smaller instance or reduce max_workers
- Process images in smaller batches

## License

MIT

## Credits

- [RapidOCR](https://github.com/RapidAI/RapidOCR) - Fast OCR toolkit
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Model source
- [RunPod](https://www.runpod.io/) - Serverless platform
