from app.intelligence.event_analyzer import EventAnalyzer


def test_gym_equipment_takes_priority_over_weak_music_detection():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {
                "dumbbell": 12,
                "weight plate": 8,
                "drum": 1,
                "car": 3,
            },
        }
    )

    assert result["event_type"] == "fitness / physical activity"
    assert result["category"] == "action"
    assert "strength training or gym workout" in result["actions"]
    assert "dumbbell" in result["detected_objects"]


def test_single_drum_does_not_classify_as_music():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {
                "drum": 1,
                "chair": 5,
            },
        }
    )

    assert result["event_type"] != "music / performance"
