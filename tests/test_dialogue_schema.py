from app.intelligence.event_analyzer import EventAnalyzer
from app.intelligence.scene_intelligence import SceneIntelligence


def test_dialogue_type_is_unknown_without_vision_enrichment():
    result = SceneIntelligence(
        EventAnalyzer()
    ).analyze_scene(
        {
            "scene_number": 1,
            "maximum_people": 2,
            "objects": {},
        }
    )

    assert result["event_type"] == "conversation / interaction"
    assert result["dialogue_type"] == "unknown"
    assert result["dialogue_evidence"] == []
