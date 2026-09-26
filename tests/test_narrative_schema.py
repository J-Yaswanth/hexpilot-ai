from app.intelligence.event_analyzer import EventAnalyzer
from app.intelligence.scene_intelligence import SceneIntelligence


def test_narrative_type_is_unknown_without_vision_enrichment():
    result = SceneIntelligence(
        EventAnalyzer()
    ).analyze_scene(
        {
            "scene_number": 1,
            "maximum_people": 1,
            "objects": {},
        }
    )

    assert result["narrative_type"] == "unknown"
    assert result["narrative_evidence"] == []
