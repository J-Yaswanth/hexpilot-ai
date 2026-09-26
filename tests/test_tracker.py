import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from app.vision.tracker import ObjectTracker


# ============================================================
# CREATE TRACKER
# ============================================================

tracker = ObjectTracker()


# ============================================================
# SIMULATED YOLO TRACKING RESULTS
# ============================================================

detections_frame_1 = [

    {
        "label": "person",
        "confidence": 0.92,
        "bbox": [100, 100, 200, 400],
        "track_id": 1
    },

    {
        "label": "person",
        "confidence": 0.89,
        "bbox": [300, 120, 400, 410],
        "track_id": 2
    },

    {
        "label": "person",
        "confidence": 0.94,
        "bbox": [500, 110, 600, 420],
        "track_id": 3
    },

    {
        "label": "person",
        "confidence": 0.91,
        "bbox": [700, 130, 800, 430],
        "track_id": 4
    }
]


# ============================================================
# UPDATE TRACKER
# ============================================================

tracker.update(
    detections_frame_1
)


# ============================================================
# SIMULATE SAME PEOPLE IN ANOTHER FRAME
# ============================================================

detections_frame_2 = [

    {
        "label": "person",
        "confidence": 0.93,
        "bbox": [105, 102, 205, 402],
        "track_id": 1
    },

    {
        "label": "person",
        "confidence": 0.90,
        "bbox": [305, 122, 405, 412],
        "track_id": 2
    },

    {
        "label": "person",
        "confidence": 0.95,
        "bbox": [505, 112, 605, 422],
        "track_id": 3
    },

    {
        "label": "person",
        "confidence": 0.92,
        "bbox": [705, 132, 805, 432],
        "track_id": 4
    }
]


tracker.update(
    detections_frame_2
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 50)
print("AI OBJECT TRACKER TEST")
print("=" * 50)

print()

unique_counts = tracker.get_unique_counts()

for label, count in unique_counts.items():

    print(
        f"{label.title()} "
        f"→ {count} unique object(s)"
    )

print()

all_tracks = tracker.get_all_tracks()

print(
    f"Total unique tracked objects: "
    f"{len(all_tracks)}"
)

print()

print("TRACK DETAILS")
print("-" * 50)

for track in all_tracks:

    print(
        f"ID: {track['track_id']} | "
        f"Object: {track['label']} | "
        f"Confidence: {track['confidence']:.2f}"
    )

print()

print("Tracker test completed successfully.")