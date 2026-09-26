from app.intelligence.event_analyzer import EventAnalyzer


def test_event_result_contains_evidence_confidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {
                "dumbbell": 10,
                "weight plate": 8,
            },
        }
    )

    assert 0.0 <= result["evidence_confidence"] <= 1.0
