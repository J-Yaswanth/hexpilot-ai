import sys
from pathlib import Path

# Allow imports from project root
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import cv2

from app.vision.open_vocabulary import (
    OpenVocabularyDetector
)


# -----------------------------
# Video path
# -----------------------------

video_path = "data/videos/test.mp4"


# -----------------------------
# Load detector
# -----------------------------

print("\nLoading Open-Vocabulary AI...")
detector = OpenVocabularyDetector()

print("Model loaded successfully!")


# -----------------------------
# Open video
# -----------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("Could not open video.")

    exit()


# -----------------------------
# Read first frame
# -----------------------------

ret, frame = cap.read()

if not ret:

    print("Could not read frame.")

    cap.release()

    exit()


# -----------------------------
# Detect objects
# -----------------------------

print("\nAnalyzing frame...")

detections = detector.detect(frame)


# -----------------------------
# Display results
# -----------------------------

print("\nAI OBJECT DETECTIONS")
print("=" * 50)


if not detections:

    print("No objects detected.")

else:

    for detection in detections:

        print(
            f"{detection['label']} "
            f"| Confidence: "
            f"{detection['confidence']:.2f} "
            f"| BBox: "
            f"{detection['bbox']}"
        )


# -----------------------------
# Cleanup
# -----------------------------

cap.release()

print("\nTest completed.")