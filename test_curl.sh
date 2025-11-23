#!/bin/bash
# Quick curl test for RapidOCR Load Balancer endpoint

RAPIDOCR_ENDPOINT_URL="${RAPIDOCR_ENDPOINT_URL:-https://your-endpoint-id.api.runpod.ai}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:-your_api_key_here}"

echo "Testing endpoint: $RAPIDOCR_ENDPOINT_URL"
echo ""

# Create a simple test image base64
TEST_IMAGE=$(python3 -c "
from PIL import Image, ImageDraw
import base64
from io import BytesIO

# Create test image with text
img = Image.new('RGB', (400, 100), color='white')
draw = ImageDraw.Draw(img)
draw.text((10, 40), 'Test OCR 123', fill='black')

# Convert to base64
buffered = BytesIO()
img.save(buffered, format='PNG')
print(base64.b64encode(buffered.getvalue()).decode())
")

echo "Generated test image (${#TEST_IMAGE} bytes base64)"
echo ""

# Test 1: Health check
echo "=== Test 1: Health Check ==="
curl -v "$RAPIDOCR_ENDPOINT_URL/ping" 2>&1 | grep -E "(< HTTP|status|healthy)"
echo ""

# Test 2: OCR request (with optional auth)
echo "=== Test 2: OCR Request ==="
if [ -n "$RUNPOD_API_KEY" ]; then
    echo "Using RunPod API key for authentication"
    curl -v -X POST "$RAPIDOCR_ENDPOINT_URL" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -d "{\"images\": [\"$TEST_IMAGE\"]}" \
      2>&1 | head -30
else
    echo "No API key set (trying without auth)"
    curl -v -X POST "$RAPIDOCR_ENDPOINT_URL" \
      -H "Content-Type: application/json" \
      -d "{\"images\": [\"$TEST_IMAGE\"]}" \
      2>&1 | head -30
fi

echo ""
echo "Done!"
