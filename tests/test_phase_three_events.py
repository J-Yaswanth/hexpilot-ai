from app.intelligence.event_analyzer import EventAnalyzer


def test_phone_use_is_classified_as_technology_use():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"smartphone": 1},
        }
    )

    assert result["event_type"] == "technology use"


def test_vehicle_scene_remains_transportation():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"car": 10},
        }
    )

    assert result["category"] == "transportation"
