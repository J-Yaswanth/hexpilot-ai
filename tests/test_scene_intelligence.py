from app.intelligence.scene_intelligence import (
    SceneIntelligence
)

from app.intelligence.event_analyzer import (
    EventAnalyzer
)


print("=" * 70)
print("AI VIDEO INTELLIGENCE STUDIO")
print("SCENE INTELLIGENCE INTEGRATION TEST")
print("=" * 70)


# ---------------------------------------------------------
# CREATE EVENT ANALYZER
# ---------------------------------------------------------

event_analyzer = EventAnalyzer()


# ---------------------------------------------------------
# CREATE SCENE INTELLIGENCE
# ---------------------------------------------------------

intelligence = SceneIntelligence(
    event_analyzer
)


# ---------------------------------------------------------
# TEST SCENE
# ---------------------------------------------------------

scene = {

    "scene_number": 1,

    "start_time": 0.04,

    "end_time": 11.04,

    "duration": 11.00,

    "maximum_people": 4,

    "average_people": 2.93,

    "objects": {

        "person": 82,

        "microphone": 12,

        "piano": 8,

        "drum": 10,
    },
}


# ---------------------------------------------------------
# ANALYZE
# ---------------------------------------------------------

result = intelligence.analyze_scene(
    scene
)


# ---------------------------------------------------------
# PRINT RESULT
# ---------------------------------------------------------

print()
print("=" * 70)
print("FINAL SCENE INTELLIGENCE")
print("=" * 70)

print()

print(
    f"Scene #{result['scene_number']}"
)

print(
    f"Time: "
    f"{result['start_time']:.2f}s "
    f"→ "
    f"{result['end_time']:.2f}s"
)

print()

print(
    f"People: "
    f"{result['people']}"
)

print()

print(
    f"Event type: "
    f"{result['event_type']}"
)

print()

print("Detected objects:")

for object_name in result[
    "detected_objects"
]:

    print(
        f"  - {object_name}"
    )

print()

print("Events:")

for event in result[
    "events"
]:

    print(
        f"  - {event}"
    )

print()

print("SCENE DESCRIPTION")
print("-" * 70)

print(
    result[
        "scene_description"
    ]
)

print()

print("=" * 70)
print("SCENE INTELLIGENCE TEST COMPLETED")
print("=" * 70)