import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import cv2

from app.vision.detector import VideoObjectDetector
from app.vision.deep_sort_tracker import DeepSortPersonTracker


VIDEO_PATH = "data/videos/test.mp4"


print("=" * 70)
print("YOLO11 + DEEPSORT PERSON TRACKING TEST")
print("=" * 70)


# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

print("\nLoading YOLO11...")

detector = VideoObjectDetector()

print("YOLO11 loaded.")


# --------------------------------------------------
# Load DeepSORT
# --------------------------------------------------

print("\nLoading DeepSORT...")

tracker = DeepSortPersonTracker()

print("DeepSORT loaded.")


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\nERROR: Could not open video.")

    sys.exit(1)


frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(
    f"\nVideo frames : {frame_count}"
)

print(
    f"Video FPS    : {fps:.2f}"
)


# --------------------------------------------------
# Tracking statistics
# --------------------------------------------------

max_people = 0

all_track_ids = set()

track_hits = {}

frame_number = 0


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # Process every 5th frame
    if frame_number % 5 != 0:
        continue


    # --------------------------------------------------
    # YOLO detection
    # --------------------------------------------------

    detections = detector.detect(
        frame,
        confidence=0.50
    )


    # --------------------------------------------------
    # DeepSORT tracking
    # --------------------------------------------------

    tracks = tracker.update(
        detections,
        frame
    )


    person_count = len(tracks)


    max_people = max(
        max_people,
        person_count
    )


    # --------------------------------------------------
    # Store track IDs
    # --------------------------------------------------

    current_ids = []

    for track in tracks:

        track_id = track["track_id"]

        current_ids.append(
            track_id
        )

        all_track_ids.add(
            track_id
        )

        if track_id not in track_hits:

            track_hits[track_id] = 0

        track_hits[track_id] += 1


    print(
        f"Frame {frame_number:3d} "
        f"→ persons: {person_count} "
        f"| IDs: {current_ids}"
    )


# --------------------------------------------------
# Release video
# --------------------------------------------------

cap.release()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("DEEPSORT TRACKING RESULTS")
print("=" * 70)


print(
    f"\nMaximum simultaneous people : {max_people}"
)

print(
    f"Total track IDs generated  : {len(all_track_ids)}"
)

print(
    f"Track IDs                   : "
    f"{sorted(all_track_ids)}"
)


print("\n")
print("TRACK HIT COUNTS")
print("-" * 70)


for track_id, hits in sorted(
    track_hits.items()
):

    print(
        f"ID {track_id} → "
        f"{hits} analyzed frames"
    )


print("\n")
print("DeepSORT tracking test completed.")