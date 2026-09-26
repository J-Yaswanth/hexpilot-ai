from pathlib import Path

from app.vision.video_pipeline import VideoAnalysisPipeline


print("=" * 70)
print("AI VIDEO INTELLIGENCE STUDIO")
print("YOLO11 + DEEPSORT FULL VIDEO PIPELINE TEST")
print("=" * 70)


# ------------------------------------------------------------
# VIDEO PATH
# ------------------------------------------------------------

video_path = Path("data/videos/test.mp4")


if not video_path.exists():
    raise FileNotFoundError(
        f"Test video not found: {video_path}"
    )


print(f"\nInput video : {video_path}")


# ------------------------------------------------------------
# CREATE PIPELINE
# ------------------------------------------------------------

pipeline = VideoAnalysisPipeline(
    model_path="yolo11n.pt",
    confidence=0.50,
    output_dir="data/outputs",
)


# ------------------------------------------------------------
# ANALYZE VIDEO
# ------------------------------------------------------------

results = pipeline.analyze(
    video_path=video_path,
    output_name="test_processed.mp4",
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL PIPELINE RESULTS")
print("=" * 70)


video = results["video"]
analysis = results["analysis"]


print("\nVIDEO")
print("-" * 70)

print(
    f"Resolution       : "
    f"{video['width']} x {video['height']}"
)

print(
    f"FPS              : "
    f"{video['fps']:.2f}"
)

print(
    f"Total frames     : "
    f"{video['total_frames']}"
)

print(
    f"Duration         : "
    f"{video['duration']:.2f} sec"
)


print("\nANALYSIS")
print("-" * 70)

print(
    f"Frames processed : "
    f"{analysis['frames_processed']}"
)

print(
    f"Maximum people   : "
    f"{analysis['max_simultaneous_people']}"
)

print(
    f"Unique identities: "
    f"{analysis['total_unique_track_ids']}"
)


print("\nTRACKS")
print("-" * 70)


for track in results["tracks"]:

    print(
        f"Person #{track['track_id']} | "
        f"Frames: {track['frames_tracked']} | "
        f"First: {track['first_frame']} | "
        f"Last: {track['last_frame']} | "
        f"Avg confidence: "
        f"{track['average_confidence']:.2f}"
    )


print("\nOUTPUT VIDEO")
print("-" * 70)

print(results["output_video"])


# ------------------------------------------------------------
# VERIFY OUTPUT
# ------------------------------------------------------------

output_path = Path(
    results["output_video"]
)


if output_path.exists():

    print("\n✅ Processed video created successfully!")

    print(
        f"File size: "
        f"{output_path.stat().st_size / (1024 * 1024):.2f} MB"
    )

else:

    raise RuntimeError(
        "❌ Output video was not created."
    )


print("\n")
print("=" * 70)
print("FULL VIDEO PIPELINE TEST COMPLETED")
print("=" * 70)