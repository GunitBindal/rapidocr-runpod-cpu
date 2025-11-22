#!/usr/bin/env python3
"""
Test RapidOCR Load Balancer Endpoint
Quick test script for RunPod Load Balancing endpoints
"""
import argparse
import base64
import json
import time
import os
from pathlib import Path
from io import BytesIO
import requests
from PIL import Image

try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support disabled.")
    print("Install with: pip install pdf2image")

# Configuration
ENDPOINT_URL = os.environ.get("RAPIDOCR_ENDPOINT_URL", "")

if not ENDPOINT_URL:
    print("Error: RAPIDOCR_ENDPOINT_URL environment variable not set!")
    print("Set it with: export RAPIDOCR_ENDPOINT_URL=https://your-endpoint-url.runpod.net")
    exit(1)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_images(file_path):
    """Extract images from PDF or load image file"""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    # PDF handling
    if suffix == '.pdf':
        if not PDF_SUPPORT:
            raise RuntimeError("PDF support not available. Install pdf2image: pip install pdf2image")
        print(f"📄 Converting PDF to images...")
        images = pdf2image.convert_from_path(str(file_path))
        print(f"✓ Extracted {len(images)} pages from PDF")
        return images

    # Image handling
    elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
        print(f"🖼️  Loading image file...")
        img = Image.open(file_path).convert("RGB")
        print(f"✓ Loaded image: {img.size}")
        return [img]

    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP")

def test_ocr(image, page_num):
    """Send OCR request to Load Balancer endpoint"""
    try:
        # Convert to base64
        convert_start = time.time()
        img_base64 = image_to_base64(image)
        convert_time = time.time() - convert_start

        print(f"  Page {page_num}: Sending request... (conversion: {convert_time:.2f}s)")

        # Send HTTP POST request
        ocr_start = time.time()
        response = requests.post(
            ENDPOINT_URL,
            json={"images": [img_base64]},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        ocr_time = time.time() - ocr_start

        if response.status_code == 200:
            result = response.json()
            total_time = convert_time + ocr_time

            # Count text lines
            text_lines = 0
            if result.get("success") and result.get("results"):
                for img_result in result["results"]:
                    text_lines += img_result.get("total_lines", 0)

            print(f"  ✓ Page {page_num}: {text_lines} lines (OCR: {ocr_time:.2f}s, Total: {total_time:.2f}s)")

            return {
                "page": page_num,
                "success": True,
                "text_lines": text_lines,
                "convert_time": convert_time,
                "ocr_time": ocr_time,
                "total_time": total_time,
                "result": result
            }
        else:
            print(f"  ✗ Page {page_num}: HTTP {response.status_code} - {response.text}")
            return {
                "page": page_num,
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }

    except Exception as e:
        print(f"  ✗ Page {page_num}: Error - {e}")
        return {
            "page": page_num,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test RapidOCR Load Balancer Endpoint")
    parser.add_argument("input_file", help="Path to PDF or image file")
    parser.add_argument("--output", default="test_results.json", help="Output JSON file (default: test_results.json)")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 RapidOCR Load Balancer Test")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"Input file: {args.input_file}")
    print("=" * 60)

    # Extract images
    start_time = time.time()
    images = extract_images(args.input_file)

    print(f"\n📊 Processing {len(images)} page(s) sequentially...\n")

    # Process pages sequentially (load balancer handles distribution)
    results = []
    total_text_lines = 0
    successful = 0
    failed = 0

    for i, img in enumerate(images):
        result = test_ocr(img, i + 1)
        results.append(result)

        if result["success"]:
            successful += 1
            total_text_lines += result.get("text_lines", 0)
        else:
            failed += 1

    total_time = time.time() - start_time

    # Calculate stats
    avg_ocr_time = sum(r.get("ocr_time", 0) for r in results if r["success"]) / max(successful, 1)
    avg_total_time = sum(r.get("total_time", 0) for r in results if r["success"]) / max(successful, 1)

    # Save results
    output_data = {
        "endpoint": ENDPOINT_URL,
        "input_file": str(args.input_file),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {
            "total_pages": len(images),
            "successful": successful,
            "failed": failed,
            "total_text_lines": total_text_lines,
            "total_time": total_time,
            "avg_ocr_time": avg_ocr_time,
            "avg_total_time": avg_total_time,
            "pages_per_second": len(images) / total_time if total_time > 0 else 0
        },
        "results": results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("📈 TEST SUMMARY")
    print("=" * 60)
    print(f"Total pages:          {len(images)}")
    print(f"Successful:           {successful}")
    print(f"Failed:               {failed}")
    print(f"Total text lines:     {total_text_lines}")
    print(f"")
    print(f"Total time:           {total_time:.2f}s")
    print(f"Avg OCR time:         {avg_ocr_time:.2f}s/page")
    print(f"Avg total time:       {avg_total_time:.2f}s/page")
    print(f"Pages per second:     {len(images) / total_time:.2f}")
    print("=" * 60)
    print(f"\n✓ Results saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
