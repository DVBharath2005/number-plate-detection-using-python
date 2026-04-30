# ── Windows UTF-8 fix (MUST be first) ──────────────────────────────────────
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# ────────────────────────────────────────────────────────────────────────────

"""
Vehicle Number Plate Detection and Recognition
==============================================
Uses OpenCV for plate detection and EasyOCR for text recognition.

Install dependencies:
    pip install opencv-python easyocr numpy

Usage:
    python plate_detection.py                        # Webcam
    python plate_detection.py --image car.jpg        # Image file
    python plate_detection.py --video traffic.mp4    # Video file
"""

import cv2
import numpy as np
import easyocr
import argparse
import re
import os
from datetime import datetime


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "min_plate_area": 1500,
    "max_plate_area": 80000,
    "aspect_ratio_min": 1.5,
    "aspect_ratio_max": 6.0,
    "ocr_confidence": 0.3,
    "padding": 10,
    "gpu": False,
}


# ─────────────────────────────────────────────
#  Patch EasyOCR progress bar for Windows
# ─────────────────────────────────────────────
def patch_easyocr_progress():
    """Replace EasyOCR Unicode progress bar with ASCII-safe version."""
    try:
        import easyocr.utils as eu

        def safe_reporthook(blocknum, block_size, total_size):
            downloaded = blocknum * block_size
            if total_size > 0:
                pct = min(int(downloaded * 100 / total_size), 100)
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                print(f"\rDownloading |{bar}| {pct}%  ", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\rDownloaded {mb:.1f} MB  ", end="", flush=True)

        # Patch urlretrieve calls inside easyocr.utils
        import urllib.request

        original_urlretrieve = urllib.request.urlretrieve

        def safe_urlretrieve(url, filename=None, reporthook=None, data=None):
            # Always replace the reporthook with our safe version
            return original_urlretrieve(url, filename, safe_reporthook, data)

        urllib.request.urlretrieve = safe_urlretrieve
        print("[INFO] Download progress patched for Windows.")
    except Exception as e:
        print(f"[WARN] Patch skipped: {e}")


patch_easyocr_progress()


# ─────────────────────────────────────────────
#  Plate Detector (OpenCV)
# ─────────────────────────────────────────────
class PlateDetector:
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plates = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                aspect = w / float(h) if h > 0 else 0

                if (
                    CONFIG["min_plate_area"] < area < CONFIG["max_plate_area"]
                    and CONFIG["aspect_ratio_min"] < aspect < CONFIG["aspect_ratio_max"]
                ):
                    plates.append((x, y, w, h))

        return plates

    def extract_roi(self, frame, bbox):
        x, y, w, h = bbox
        pad = CONFIG["padding"]
        hf, wf = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(wf, x + w + pad)
        y2 = min(hf, y + h + pad)
        return frame[y1:y2, x1:x2]


# ─────────────────────────────────────────────
#  OCR Reader (EasyOCR)
# ─────────────────────────────────────────────
class PlateReader:
    def __init__(self):
        print("[INFO] Loading OCR model (first run downloads ~100 MB)...")
        self.reader = easyocr.Reader(["en"], gpu=CONFIG["gpu"], verbose=False)
        print("\n[INFO] OCR model ready.")

    def read(self, roi):
        results = self.reader.readtext(roi)
        output = []
        for bbox, text, conf in results:
            if conf >= CONFIG["ocr_confidence"]:
                cleaned = self._clean(text)
                if cleaned:
                    output.append({"text": cleaned, "confidence": conf, "bbox": bbox})
        return output

    @staticmethod
    def _clean(text):
        return re.sub(r"[^A-Z0-9\-]", "", text.upper().strip())


# ─────────────────────────────────────────────
#  Drawing Helpers
# ─────────────────────────────────────────────
def draw_plate(frame, bbox, texts):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = " | ".join(
        f"{r['text']} ({r['confidence']:.0%})" for r in texts
    ) if texts else "Detected"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    banner_y = max(0, y - th - 8)
    cv2.rectangle(frame, (x, banner_y), (x + tw + 8, y), (0, 255, 0), -1)
    cv2.putText(frame, label, (x + 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def draw_info(frame, plate_count, fps=0):
    info = f"Plates: {plate_count}  FPS: {fps:.1f}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    return frame


# ─────────────────────────────────────────────
#  Core Pipeline
# ─────────────────────────────────────────────
def process_frame(frame, detector, reader):
    detections = []
    plates = detector.detect(frame)
    for bbox in plates:
        roi = detector.extract_roi(frame, bbox)
        texts = reader.read(roi)
        draw_plate(frame, bbox, texts)
        detections.append({"bbox": bbox, "readings": texts})
    return frame, detections


# ─────────────────────────────────────────────
#  Run Modes
# ─────────────────────────────────────────────
def run_image(path, detector, reader):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot open image: {path}")
        sys.exit(1)

    annotated, detections = process_frame(frame, detector, reader)

    print(f"\n{'='*40}")
    print(f"Image : {path}")
    print(f"Plates found: {len(detections)}")
    for i, d in enumerate(detections, 1):
        for r in d["readings"]:
            print(f"  [{i}] {r['text']}  ({r['confidence']:.0%})")
    print(f"{'='*40}\n")

    out_path = f"result_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"[INFO] Saved -> {out_path}")

    cv2.imshow("Plate Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(source, detector, reader):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        sys.exit(1)

    fps_timer = datetime.now()
    fps = 0
    frame_count = 0

    print("[INFO] Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        annotated, detections = process_frame(frame, detector, reader)

        elapsed = (datetime.now() - fps_timer).total_seconds()
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = datetime.now()

        draw_info(annotated, len(detections), fps)
        cv2.imshow("Plate Detection (Q to quit)", annotated)

        for d in detections:
            for r in d["readings"]:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Detected: {r['text']}  ({r['confidence']:.0%})")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Number Plate Detection")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", "-i", help="Path to image file")
    group.add_argument("--video", "-v", help="Path to video file")
    args = parser.parse_args()

    detector = PlateDetector()
    reader = PlateReader()

    if args.image:
        run_image(args.image, detector, reader)
    elif args.video:
        run_video(args.video, detector, reader)
    else:
        print("[INFO] Starting webcam (device 0)...")
        run_video(0, detector, reader)


if __name__ == "__main__":
    main()
