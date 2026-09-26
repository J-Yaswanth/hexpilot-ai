import cv2
from ultralytics import YOLO

from app.vision.deep_sort_tracker import (
    DeepSortPersonTracker
)


VIDEO_PATH = "data/videos/test.mp4"
MODEL_PATH = "yolo11n.pt"


print("=" * 70)
print("YOLO11 + DEEPSORT INTEGRATED PERSON ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# LOAD YOLO
# --------------------------------------------------

print("\nLoading YOLO11...")

model = YOLO(MODEL_PATH)

print("YOLO11 loaded.")


# --------------------------------------------------
# LOAD DEEPSORT
# --------------------------------------------------

print("\nLoading DeepSORT...")

tracker = DeepSortPersonTracker()

print("DeepSORT loaded.")


# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("\nERROR: Could not open video.")

    raise SystemExit


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
# ANALYSIS VARIABLES
# --------------------------------------------------

frame_number = 0

all_track_ids = set()

track_history = {}

max_simultaneous_people = 0


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # Analyze every 5th frame
    if frame_number % 5 != 0:
        continue


    # --------------------------------------------------
    # YOLO PERSON DETECTION
    # --------------------------------------------------

    results = model(
        frame,
        conf=0.50,
        classes=[0],
        verbose=False
    )


    person_detections = []


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            # COCO class 0 = person
            if class_id != 0:
                continue


            confidence = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            person_detections.append(
                {
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                    "confidence": confidence
                }
            )


    # --------------------------------------------------
    # DEEPSORT
    # --------------------------------------------------

    tracks = tracker.update(
        person_detections,
        frame
    )


    current_ids = []


    for track in tracks:

        track_id = track["track_id"]

        current_ids.append(
            track_id
        )

        all_track_ids.add(
            track_id
        )


        if track_id not in track_history:

            track_history[
                track_id
            ] = {
                "frames": 0,
                "first_frame": frame_number,
                "last_frame": frame_number
            }


        track_history[
            track_id
        ]["frames"] += 1


        track_history[
            track_id
        ]["last_frame"] = frame_number


    # --------------------------------------------------
    # MAXIMUM SIMULTANEOUS PEOPLE
    # --------------------------------------------------

    current_people = len(
        current_ids
    )


    max_simultaneous_people = max(
        max_simultaneous_people,
        current_people
    )


    print(
        f"Frame {frame_number:3d} "
        f"→ detections: {len(person_detections)} "
        f"| confirmed people: {current_people} "
        f"| IDs: {current_ids}"
    )


# --------------------------------------------------
# RELEASE VIDEO
# --------------------------------------------------

cap.release()


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n")
print("=" * 70)
print("INTEGRATED TRACKING RESULTS")
print("=" * 70)


print(
    f"\nMaximum simultaneous people : "
    f"{max_simultaneous_people}"
)


print(
    f"Total DeepSORT identities    : "
    f"{len(all_track_ids)}"
)


print(
    f"Track IDs                    : "
    f"{sorted(all_track_ids)}"
)


# --------------------------------------------------
# TRACK DETAILS
# --------------------------------------------------

print("\n")
print("TRACK DETAILS")
print("-" * 70)


for track_id in sorted(
    track_history
):

    info = track_history[
        track_id
    ]


    print(
        f"Person #{track_id} | "
        f"Frames tracked: {info['frames']} | "
        f"First frame: {info['first_frame']} | "
        f"Last frame: {info['last_frame']}"
    )


print("\n")
print("YOLO11 + DeepSORT integration test completed.")