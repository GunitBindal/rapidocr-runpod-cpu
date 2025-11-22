#!/usr/bin/env python3
"""
Batch OCR processing with RapidOCR on RunPod CPU
Supports PDF and image files with concurrent requests
"""
import argparse
import base64
import json
import time
import os
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from PIL import Image

try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support disabled.")
    print("Install with: pip install pdf2image")

# Configuration - Get from environment variable
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY environment variable not set!")
    print("Set it with: export RUNPOD_API_KEY=your_key_here")
    exit(1)

if not ENDPOINT_ID:
    print("Error: RUNPOD_ENDPOINT_ID environment variable not set!")
    print("Set it with: export RUNPOD_ENDPOINT_ID=your_endpoint_id")
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

def submit_ocr_job(image_base64, page_num):
    """Submit OCR job to RunPod"""
    response = requests.post(
        RUNPOD_URL,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "input": {
                "images": [image_base64]
            }
        },
        timeout=30
    )

    result = response.json()
    return {
        "page": page_num,
        "job_id": result["id"],
        "status": result["status"],
        "submit_time": time.time()
    }

def check_job_status(job_id):
    """Check status of a job"""
    response = requests.get(
        f"{STATUS_URL}/{job_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30
    )
    return response.json()

def wait_for_job(job_id, max_wait=300, poll_interval=2):
    """Wait for job to complete"""
    start = time.time()

    while time.time() - start < max_wait:
        result = check_job_status(job_id)
        status = result.get("status")

        if status == "COMPLETED":
            return result
        elif status == "FAILED":
            raise RuntimeError(f"Job failed: {result.get('error', 'Unknown error')}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")

def process_single_page(page_num, image, stats):
    """Process a single page"""
    try:
        # Convert to base64
        convert_start = time.time()
        img_base64 = image_to_base64(image)
        convert_time = time.time() - convert_start

        print(f"  Page {page_num}: Submitting job... (conversion: {convert_time:.2f}s)")

        # Submit job
        submit_start = time.time()
        job_info = submit_ocr_job(img_base64, page_num)
        submit_time = time.time() - submit_start

        print(f"  ✓ Page {page_num}: Job {job_info['job_id']} submitted")

        # Return job info without waiting
        return {
            "page": page_num,
            "job_id": job_info["job_id"],
            "convert_time": convert_time,
            "submit_time": submit_time,
            "submit_timestamp": time.time()
        }

    except Exception as e:
        print(f"  ✗ Page {page_num}: Submission error - {e}")
        return {
            "page": page_num,
            "error": str(e),
            "submit_time": 0,
            "convert_time": 0
        }

def wait_for_results(job_submissions, stats):
    """Wait for all submitted jobs to complete"""
    results = []

    print(f"\n⏳ Waiting for {len(job_submissions)} jobs to complete...\n")

    for submission in job_submissions:
        if "error" in submission:
            results.append(submission)
            stats["failed"] += 1
            continue

        page_num = submission["page"]
        job_id = submission["job_id"]

        try:
            wait_start = time.time()
            result = wait_for_job(job_id)
            wait_time = time.time() - wait_start

            total_time = time.time() - submission["submit_timestamp"]

            print(f"  ✓ Page {page_num}: Complete (OCR: {wait_time:.2f}s, Total: {total_time:.2f}s)")

            # Update stats
            stats["total_conversion_time"] += submission["convert_time"]
            stats["total_submit_time"] += submission["submit_time"]
            stats["total_wait_time"] += wait_time
            stats["total_processing_time"] += total_time
            stats["completed"] += 1

            results.append({
                "page": page_num,
                "job_id": job_id,
                "result": result.get("output"),
                "timings": {
                    "conversion": submission["convert_time"],
                    "submit": submission["submit_time"],
                    "ocr": wait_time,
                    "total": total_time
                }
            })

        except Exception as e:
            print(f"  ✗ Page {page_num}: Error - {e}")
            stats["failed"] += 1
            results.append({
                "page": page_num,
                "job_id": job_id,
                "error": str(e)
            })

    return results

def main():
    parser = argparse.ArgumentParser(description="Batch OCR processing with RapidOCR on CPU")
    parser.add_argument("input_file", help="Path to PDF or image file")
    parser.add_argument("--output-dir", default="ocr_output", help="Output directory (default: ocr_output)")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent requests (default: 5)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("🚀 RapidOCR CPU Batch Processing")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Max workers: {args.max_workers}")
    print("=" * 60)

    # Extract images
    start_time = time.time()
    images = extract_images(args.input_file)
    extraction_time = time.time() - start_time

    print(f"\n📊 Processing {len(images)} page(s) with {args.max_workers} concurrent workers...\n")

    # Statistics
    stats = {
        "total_pages": len(images),
        "completed": 0,
        "failed": 0,
        "total_conversion_time": 0,
        "total_submit_time": 0,
        "total_wait_time": 0,
        "total_processing_time": 0,
        "extraction_time": extraction_time
    }

    # Phase 1: Submit all jobs concurrently
    print("📤 Phase 1: Submitting all jobs concurrently...\n")
    job_submissions = []
    process_start = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_page, i+1, img, stats): i+1
            for i, img in enumerate(images)
        }

        for future in as_completed(futures):
            submission = future.result()
            job_submissions.append(submission)

    submission_time = time.time() - process_start
    print(f"\n✓ All jobs submitted in {submission_time:.2f}s")

    # Phase 2: Wait for all results
    results = wait_for_results(job_submissions, stats)

    total_time = time.time() - start_time

    # Sort results by page number
    results.sort(key=lambda x: x["page"])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"

    output_data = {
        "metadata": {
            "input_file": str(args.input_file),
            "timestamp": timestamp,
            "total_pages": len(images),
            "max_workers": args.max_workers,
            "engine": "RapidOCR CPU (PP-OCRv4)"
        },
        "statistics": {
            **stats,
            "total_time": total_time,
            "avg_time_per_page": stats["total_processing_time"] / len(images) if images else 0,
            "pages_per_second": len(images) / total_time if total_time > 0 else 0
        },
        "results": results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Save text output
    text_file = output_dir / f"text_{timestamp}.txt"
    with open(text_file, 'w', encoding='utf-8') as f:
        for result in results:
            if "error" not in result and result.get("result", {}).get("success"):
                f.write(f"\n{'='*60}\n")
                f.write(f"Page {result['page']}\n")
                f.write(f"{'='*60}\n\n")

                for img_result in result["result"]["results"]:
                    for text_line in img_result.get("text_lines", []):
                        f.write(f"{text_line['text']}\n")

    # Print summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total pages:          {stats['total_pages']}")
    print(f"Completed:            {stats['completed']}")
    print(f"Failed:               {stats['failed']}")
    print(f"")
    print(f"Image extraction:     {stats['extraction_time']:.2f}s")
    print(f"Total processing:     {total_time:.2f}s")
    print(f"Avg time per page:    {stats['total_processing_time'] / len(images):.2f}s")
    print(f"Pages per second:     {len(images) / total_time:.2f}")
    print(f"")
    print(f"Avg conversion time:  {stats['total_conversion_time'] / len(images):.2f}s")
    print(f"Avg OCR time:         {stats['total_wait_time'] / len(images):.2f}s")
    print("=" * 60)
    print(f"\n✓ Results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {text_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
