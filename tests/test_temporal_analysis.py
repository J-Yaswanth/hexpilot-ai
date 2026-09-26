import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import cv2

from ultralytics import YOLO

from app.vision.temporal_analysis import (
    TemporalPersonAnalyzer
)


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "data/videos/test.mp4"

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.50

FRAME_INTERVAL = 10


# ============================================================
# START
# ============================================================

print()

print("=" * 70)

print(
    "AI TEMPORAL PERSON ANALYSIS TEST"
)

print("=" * 70)

print()


# ============================================================
# LOAD MODEL
# ============================================================

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


# ============================================================
# CREATE ANALYZER
# ============================================================

analyzer = TemporalPersonAnalyzer(
    min_observations=3,
    confidence_threshold=CONFIDENCE
)


# ============================================================
# OPEN VIDEO
# ============================================================

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


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()

    if not ret:

        break

    frame_number += 1


    # --------------------------------------------------------
    # Analyze every Nth frame
    # --------------------------------------------------------

    if frame_number % FRAME_INTERVAL != 0:

        continue


    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

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

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[
                class_id
            ]


            if class_name != "person":

                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            detections.append({

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
                    ]

            })


    # --------------------------------------------------------
    # Send frame to temporal analyzer
    # --------------------------------------------------------

    analyzer.add_frame(
        detections
    )


    print(
        f"Frame {frame_number:3d}"
        f" → {len(detections)} person(s)"
    )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# RESULTS
# ============================================================

summary = analyzer.get_summary()

statistics = (
    summary[
        "detection_statistics"
    ]
)

presence = (
    summary[
        "person_presence"
    ]
)


print()

print("=" * 70)

print(
    "TEMPORAL ANALYSIS RESULTS"
)

print("=" * 70)

print()


print(
    "Frames analyzed       :",
    statistics[
        "frames_analyzed"
    ]
)

print(
    "Minimum detections    :",
    statistics[
        "minimum"
    ]
)

print(
    "Maximum detections    :",
    statistics[
        "maximum"
    ]
)

print(
    "Average detections    :",
    statistics[
        "average"
    ]
)

print()


print(
    "Estimated people      :",
    presence[
        "estimated_people"
    ]
)

print(
    "Temporal support      :",
    presence[
        "temporal_support"
    ]
)

print(
    "Average confidence    :",
    presence[
        "average_detection_confidence"
    ]
)

print(
    "Evidence confidence   :",
    presence[
        "confidence"
    ]
)

print(
    "Supporting frames     :",
    presence[
        "supporting_frames"
    ],
    "/",
    presence[
        "total_frames"
    ]
)


print()

print(
    "Temporal analysis completed."
)