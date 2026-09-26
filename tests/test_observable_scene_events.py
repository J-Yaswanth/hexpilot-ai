from app.intelligence.event_analyzer import EventAnalyzer


def test_cooking_scene_requires_food_preparation_evidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"pot": 2, "pan": 1},
        }
    )

    assert result["event_type"] == "cooking / food preparation"
    assert result["category"] == "daily life"


def test_eating_scene_uses_dining_evidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"plate": 2, "fork": 1, "food": 1},
        }
    )

    assert result["event_type"] == "eating / meal"


def test_multiple_people_fall_back_to_interaction():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 2,
            "objects": {},
        }
    )

    assert result["event_type"] == "conversation / interaction"
