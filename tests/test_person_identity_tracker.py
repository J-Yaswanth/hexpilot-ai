import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

import cv2

from ultralytics import YOLO

from app.vision.person_tracker import (
    PersonIdentityTracker
)


VIDEO_PATH = "data/videos/test.mp4"

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.50

FRAME_INTERVAL = 5


print()

print("=" * 70)

print(
    "AI PERSON IDENTITY TRACKER TEST"
)

print("=" * 70)

print()


print(
    "Loading YOLO11..."
)

model = YOLO(
    MODEL_PATH
)

print(
    "Model loaded."
)

print()


tracker = PersonIdentityTracker(
    max_missed_frames=8,
    min_hits=3,
    max_center_distance=180,
    min_iou=0.05,
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    print(
        "ERROR: Could not open video."
    )

    sys.exit(1)


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)


print(
    f"Video frames : {total_frames}"
)

print(
    f"Video FPS    : {fps:.2f}"
)

print()


frame_number = 0

maximum_confirmed = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1

    if frame_number % FRAME_INTERVAL != 0:

        continue


    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    detections = []


    for result in results:

        if result.boxes is None:

            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            class_name = model.names[
                class_id
            ]


            if class_name != "person":

                continue


            confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            detections.append(
                {
                    "label":
                        "person",

                    "confidence":
                        confidence,

                    "bbox":
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                }
            )


    tracked = tracker.update(
        detections
    )


    confirmed = [
        item
        for item in tracked
        if item["confirmed"]
    ]


    maximum_confirmed = max(
        maximum_confirmed,
        len(confirmed)
    )


    print(
        f"Frame {frame_number:3d}"
        f" → detections:"
        f" {len(detections)}"
        f" | confirmed tracks:"
        f" {len(confirmed)}"
    )


cap.release()


summary = tracker.get_summary()


print()

print("=" * 70)

print(
    "TRACKING RESULTS"
)

print("=" * 70)

print()


print(
    "Maximum confirmed people :",
    maximum_confirmed
)

print(
    "Currently active tracks  :",
    summary[
        "active_confirmed_tracks"
    ]
)

print(
    "Track IDs                :",
    summary[
        "track_ids"
    ]
)

print()


print(
    "TRACK DETAILS"
)

print("-" * 70)


for track in summary["tracks"]:

    print(
        f"ID: {track['track_id']}"
        f" | Hits: {track['hits']}"
        f" | Confidence:"
        f" {track['confidence']:.2f}"
        f" | Frames:"
        f" {track['first_frame']}"
        f"-"
        f"{track['last_frame']}"
    )


print()

print(
    "Person identity tracking test completed."
)