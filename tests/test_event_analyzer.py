from app.intelligence.event_analyzer import (
    EventAnalyzer
)


print("=" * 70)
print("AI VIDEO INTELLIGENCE STUDIO")
print("EVENT ANALYZER TEST")
print("=" * 70)


scene = {
    "maximum_people": 4,

    "objects": {
        "person": 82,
        "microphone": 12,
        "piano": 8,
        "drum": 10,
    }
}


print("\n")
print("=" * 70)
print("INPUT SCENE")
print("=" * 70)

print(
    f"People : "
    f"{scene['maximum_people']}"
)

print(
    "Objects:"
)

for name, count in scene["objects"].items():

    print(
        f"  - {name}: {count}"
    )


# ---------------------------------------------------------
# EVENT ANALYZER
# ---------------------------------------------------------

analyzer = EventAnalyzer()

result = analyzer.analyze_scene(
    scene
)


print("\n")
print("=" * 70)
print("EVENT ANALYSIS RESULT")
print("=" * 70)


print(
    f"\nEvent type : "
    f"{result['event_type']}"
)


print(
    f"People : "
    f"{result['people_count']}"
)


print("\nEvents:")

for event in result["events"]:

    print(
        f"  - {event}"
    )


print("\nDescription:")

print(
    result["description"]
)


print("\nDetected objects:")

for obj in result["detected_objects"]:

    print(
        f"  - {obj}"
    )


print("\n")
print("=" * 70)
print("EVENT ANALYZER TEST COMPLETED")
print("=" * 70)