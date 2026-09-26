import json
from unittest.mock import patch

from app.intelligence.vision_enrichment import VisionEnricher


def test_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_VISION_API_KEY", raising=False)
    assert VisionEnricher().enrich(b"frame", {"scene": 1}) is None


def test_valid_openai_compatible_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = {
        "choices": [{"message": {"content": json.dumps({
            "description": "A person walks past a table.",
            "category": "activity",
            "actions": ["walking"],
            "appearance": {"lighting": "daylight"},
            "objects": [{"label": "person", "count": 1}],
        })}}],
    }

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(response).encode()

    with patch("app.intelligence.vision_enrichment.request.urlopen", return_value=MockResponse()) as urlopen:
        result = VisionEnricher(base_url="http://provider.test/v1", model="vision-model").enrich(
            b"\x89PNG", {"scene_number": 2}
        )

    assert result.description == "A person walks past a table."
    assert result.category == "activity"
    assert result.actions == ["walking"]
    assert result.objects == [{"label": "person", "count": 1}]
    assert urlopen.call_args.args[0].full_url.endswith("/chat/completions")
