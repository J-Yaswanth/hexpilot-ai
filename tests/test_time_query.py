import pytest

from app.intelligence.time_query import (
    describe_time_range,
    find_scenes_in_range,
)


SCENES = [
    {
        "scene_number": 1,
        "start_time": 0.0,
        "end_time": 10.0,
        "scene_description": "People are singing.",
    },
    {
        "scene_number": 2,
        "start_time": 10.0,
        "end_time": 20.0,
        "scene_description": "People are dancing.",
    },
]


def test_find_scenes_returns_overlapping_scenes():
    matches = find_scenes_in_range(SCENES, 9.0, 11.0)

    assert [scene["scene_number"] for scene in matches] == [1, 2]


def test_describe_time_range_combines_scene_descriptions():
    result = describe_time_range(SCENES, 0.5, 9.5)

    assert result["description"] == "People are singing."


def test_find_scenes_rejects_invalid_range():
    with pytest.raises(ValueError, match="end time"):
        find_scenes_in_range(SCENES, 5, 5)
