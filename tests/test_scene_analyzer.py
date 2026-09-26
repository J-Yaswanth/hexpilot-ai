from app.intelligence.scene_analyzer import SceneAnalyzer


VIDEO_PATH = "data/videos/test.mp4"


print("=" * 70)
print("AI VIDEO INTELLIGENCE STUDIO")
print("SCENE VISUAL ANALYSIS TEST")
print("=" * 70)


# ---------------------------------------------------------
# CREATE SCENE ANALYZER
# ---------------------------------------------------------

analyzer = SceneAnalyzer(
    model_path="yolo11n.pt",
    confidence=0.50,
    sample_interval=10,
)


# ---------------------------------------------------------
# TEMPORARY SCENE INFORMATION
# ---------------------------------------------------------
#
# Our scene_detector.py already detected:
#
# Scene #1
# 0.04s → 11.04s
#
# We are passing that result into the analyzer.
# ---------------------------------------------------------

scenes = [
    {
        "scene_number": 1,
        "start_time": 0.04,
        "end_time": 11.04,
    }
]


# ---------------------------------------------------------
# ANALYZE VIDEO
# ---------------------------------------------------------

results = analyzer.analyze_video(
    video_path=VIDEO_PATH,
    scenes=scenes,
)


# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------

print()
print("=" * 70)
print("FINAL SCENE ANALYSIS RESULTS")
print("=" * 70)


for scene in results:

    print()

    print(
        f"Scene #{scene['scene_number']}"
    )

    print(
        "-" * 70
    )

    print(
        f"Time       : "
        f"{scene['start_time']:.2f}s "
        f"→ "
        f"{scene['end_time']:.2f}s"
    )

    print(
        f"Duration   : "
        f"{scene['duration']:.2f}s"
    )

    print(
        f"Sampled frames : "
        f"{scene['sampled_frames']}"
    )

    print(
        f"Maximum people : "
        f"{scene['maximum_people']}"
    )

    print(
        f"Average people : "
        f"{scene['average_people']:.2f}"
    )

    print()

    print(
        "Objects:"
    )

    if scene["objects"]:

        for object_name, count in (
            scene["objects"].items()
        ):

            print(
                f"  {object_name}: "
                f"{count}"
            )

    else:

        print(
            "  None"
        )


print()
print("=" * 70)
print("SCENE VISUAL ANALYSIS TEST COMPLETED")
print("=" * 70)