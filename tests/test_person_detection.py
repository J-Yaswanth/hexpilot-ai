import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import cv2
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = "data/videos/test.mp4"

MODEL_PATH = "yolo11n.pt"

CONFIDENCE = 0.50

FRAME_INTERVAL = 10


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("AI PERSON DETECTION AUDIT")
print("=" * 60)
print()

print("Loading YOLO11...")

model = YOLO(
    MODEL_PATH
)

print("Model loaded.")
print()


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
# STATISTICS
# ============================================================

frames_tested = 0

person_counts = []

confidence_values = []


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    # Test every Nth frame
    if frame_number % FRAME_INTERVAL != 0:

        continue


    frames_tested += 1


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    person_count = 0


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


            # Only PERSON
            if class_name != "person":

                continue


            person_count += 1

            confidence_values.append(
                confidence
            )


    person_counts.append(
        person_count
    )


    print(
        f"Frame {frame_number:4d}"
        f" → {person_count} person(s)"
    )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# FINAL ANALYSIS
# ============================================================

print()
print("=" * 60)
print("DETECTION AUDIT RESULTS")
print("=" * 60)
print()


if not person_counts:

    print(
        "No frames were analyzed."
    )

    sys.exit(0)


print(
    f"Frames tested        : "
    f"{frames_tested}"
)


print(
    f"Minimum persons/frame: "
    f"{min(person_counts)}"
)


print(
    f"Maximum persons/frame: "
    f"{max(person_counts)}"
)


print(
    f"Average persons/frame: "
    f"{sum(person_counts) / len(person_counts):.2f}"
)


if confidence_values:

    print(
        f"Minimum confidence   : "
        f"{min(confidence_values):.2f}"
    )

    print(
        f"Maximum confidence   : "
        f"{max(confidence_values):.2f}"
    )

    print(
        f"Average confidence   : "
        f"{sum(confidence_values) / len(confidence_values):.2f}"
    )


print()
print(
    "Detection audit completed."
)