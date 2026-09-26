import json
from unittest.mock import patch

from app.intelligence.event_analyzer import EventAnalyzer
from app.intelligence.scene_intelligence import SceneIntelligence
from app.intelligence.scene_taxonomy import (
    SCENE_LABEL_GROUPS,
    SCENE_TAXONOMY,
)
from app.intelligence.vision_enrichment import VisionEnricher


def test_scene_taxonomy_contains_all_requested_groups():
    assert len(SCENE_TAXONOMY) == 32
    assert sum(len(labels) for labels in SCENE_TAXONOMY.values()) == 383
    assert "Fight" in SCENE_TAXONOMY["Action scenes"]
    assert "Emotional reunion" in SCENE_TAXONOMY["Emotional scenes"]
    assert "Time-lapse" in SCENE_TAXONOMY["Special cinematic scenes"]
    assert "Factory / Industrial area" in SCENE_TAXONOMY["Location-based scenes"]
    assert "Proposal" in SCENE_TAXONOMY["Romance / Relationship"]
    assert "Jump scare" in SCENE_TAXONOMY["Horror"]
    assert "Alien encounter" in SCENE_TAXONOMY["Fantasy / Sci-Fi"]
    assert "Plot development" in SCENE_TAXONOMY["Narrative / Structural"]


def test_scene_labels_are_preserved_in_final_scene_result():
    labels = [{
        "group": "Action scenes",
        "label": "Fight",
        "confidence": 0.88,
        "evidence": ["Two people are visibly fighting."],
    }]

    result = SceneIntelligence(EventAnalyzer()).analyze_scene({
        "maximum_people": 2,
        "objects": {},
        "scene_labels": labels,
    })

    assert result["scene_labels"] == labels


def test_every_taxonomy_label_is_recognized_with_all_group_memberships():
    labels = VisionEnricher._parse_scene_labels([
        {
            "label": label,
            "confidence": 0.75,
            "evidence": ["Supported by test evidence."],
        }
        for label in SCENE_LABEL_GROUPS
    ])

    parsed_by_label = {
        item["label"].casefold(): item
        for item in labels
    }
    assert len(parsed_by_label) == len(SCENE_LABEL_GROUPS)
    for label_key, groups in SCENE_LABEL_GROUPS.items():
        assert parsed_by_label[label_key]["groups"] == groups


def test_vision_enrichment_validates_scene_labels(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "description": "People dance on a stage.",
                    "scene_labels": [
                        {
                            "label": "Dancing",
                            "confidence": 0.92,
                            "evidence": ["A person is visibly dancing."],
                        },
                        {
                            "label": "Definitely a flashback",
                            "confidence": 0.8,
                            "evidence": ["Unsupported label."],
                        },
                        {
                            "label": "Singing",
                            "confidence": 1.2,
                            "evidence": ["Invalid confidence."],
                        },
                    ],
                }),
            },
        }],
    }

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(response).encode()

    with patch(
        "app.intelligence.vision_enrichment.request.urlopen",
        return_value=MockResponse(),
    ):
        result = VisionEnricher().enrich(b"frame")

    assert result.scene_labels == [{
        "group": "Music / Performance scenes",
        "groups": ["Music / Performance scenes", "Music / Dance"],
        "label": "Dancing",
        "confidence": 0.92,
        "evidence": ["A person is visibly dancing."],
    }]
