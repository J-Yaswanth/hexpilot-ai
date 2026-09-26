from app.intelligence.event_analyzer import EventAnalyzer


def test_hospital_location_requires_medical_evidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"hospital bed": 1, "medical equipment": 2},
        }
    )

    assert result["location"] == "hospital"


def test_classroom_location_is_inferred_from_classroom_evidence():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 3,
            "objects": {"classroom": 1, "school desk": 4},
        }
    )

    assert result["location"] == "school / classroom"


def test_weak_objects_do_not_force_a_cinema_location():
    result = EventAnalyzer().analyze_scene(
        {
            "maximum_people": 1,
            "objects": {"shirt": 1},
        }
    )

    assert result["location"] == "unknown"
