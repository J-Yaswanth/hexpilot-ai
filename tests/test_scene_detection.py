from app.intelligence.scene_detector import (
    SceneDetector
)


VIDEO_PATH = "data/videos/test.mp4"


print("=" * 70)
print("AI VIDEO INTELLIGENCE STUDIO")
print("SCENE DETECTION TEST")
print("=" * 70)


# ---------------------------------------------------------
# CREATE SCENE DETECTOR
# ---------------------------------------------------------

detector = SceneDetector(
    threshold=0.35,
    min_scene_duration=1.0,
)


# ---------------------------------------------------------
# DETECT SCENES
# ---------------------------------------------------------

results = detector.detect_scenes(
    VIDEO_PATH
)


# ---------------------------------------------------------
# PRINT FINAL RESULTS
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL SCENE RESULTS")
print("=" * 70)


print(
    f"\nVideo duration : "
    f"{results['duration']:.2f} seconds"
)

print(
    f"Total scenes  : "
    f"{results['total_scenes']}"
)


print("\n")
print("SCENE TIMELINE")
print("-" * 70)


for scene in results["scenes"]:

    print(
        f"Scene #{scene['scene_id']} | "
        f"{scene['start_time']:.2f}s → "
        f"{scene['end_time']:.2f}s | "
        f"Duration: {scene['duration']:.2f}s | "
        f"Change: {scene['change_score']:.3f}"
    )


print("\n")
print("=" * 70)
print("SCENE DETECTION TEST COMPLETED")
print("=" * 70)