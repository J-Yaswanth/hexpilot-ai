from app.intelligence.event_analyzer import EventAnalyzer


def test_gym_location_is_inferred_from_equipment():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {
                "dumbbell": 3,
                "weight plate": 4,
            },
        }
    )

    assert result["location"] == "gym"


def test_vehicle_scene_is_inferred_as_road():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 0,
            "objects": {
                "car": 4,
                "motorcycle": 2,
            },
        }
    )

    assert result["location"] == "road / street"


def test_location_stays_unknown_without_evidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {},
        }
    )

    assert result["location"] == "unknown"
