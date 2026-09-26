import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/videos/test.mp4"

MODEL_PATH = "yolo11n.pt"

THRESHOLDS = [
    0.50,
    0.35,
    0.25
]

FRAME_INTERVAL = 10


print()
print("=" * 70)
print("AI PERSON DETECTION THRESHOLD TEST")
print("=" * 70)
print()


model = YOLO(MODEL_PATH)

print("Model loaded.")
print()


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    sys.exit(1)


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)


print(f"Video frames : {total_frames}")
print(f"Video FPS    : {fps:.2f}")
print()


results_by_threshold = {}


for threshold in THRESHOLDS:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    counts = []

    frame_number = 0


    print()
    print("-" * 70)
    print(
        f"CONFIDENCE THRESHOLD: {threshold}"
    )
    print("-" * 70)


    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        if frame_number % FRAME_INTERVAL != 0:
            continue


        results = model(
            frame,
            conf=threshold,
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

                class_name = model.names[
                    class_id
                ]


                if class_name != "person":
                    continue


                person_count += 1


        counts.append(
            person_count
        )


        print(
            f"Frame {frame_number:3d}"
            f" → {person_count} person(s)"
        )


    results_by_threshold[
        threshold
    ] = counts


cap.release()


print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print()


for threshold, counts in results_by_threshold.items():

    if not counts:
        continue


    average = (
        sum(counts) /
        len(counts)
    )


    print(
        f"Threshold {threshold:.2f}"
    )

    print(
        f"  Minimum : {min(counts)}"
    )

    print(
        f"  Maximum : {max(counts)}"
    )

    print(
        f"  Average : {average:.2f}"
    )

    print()


print(
    "Threshold test completed."
)