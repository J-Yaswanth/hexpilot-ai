from app.intelligence.scene_analyzer import SceneAnalyzer


def test_object_label_aliases_canonicalize_phone_variants_to_ignored_label():
    assert SceneAnalyzer.normalize_object_label("smartphone") == "cell phone"
    assert SceneAnalyzer.normalize_object_label(" Mobile   Phone ") == "cell phone"
    assert (
        SceneAnalyzer.normalize_object_label("smartphone")
        in SceneAnalyzer.IGNORED_OBJECTS
    )


def test_water_bottle_detections_need_high_confidence():
    assert not SceneAnalyzer.meets_object_confidence_threshold("bottle", 0.59)
    assert SceneAnalyzer.meets_object_confidence_threshold("water bottle", 0.60)


def test_single_frame_duplicate_detections_do_not_meet_temporal_support():
    observations = {"sports ball": 5, "car": 4}
    frames = [
        {"objects": {"sports ball": 5, "car": 1}},
        {"objects": {"car": 1}},
    ]

    SceneAnalyzer.filter_temporally_unsupported_objects(observations, frames)

    assert "sports ball" not in observations
    assert all("sports ball" not in frame["objects"] for frame in frames)
    assert observations["car"] == 4


def test_water_bottle_requires_multiple_sampled_frames():
    observations = {"water bottle": 3, "car": 3}
    frames = [
        {"objects": {"water bottle": 1, "car": 1}},
        {"objects": {"water bottle": 1, "car": 1}},
        {"objects": {"car": 1}},
    ]

    SceneAnalyzer.filter_temporally_unsupported_objects(observations, frames)

    assert "water bottle" not in observations
    assert all("water bottle" not in frame["objects"] for frame in frames)
    assert observations["car"] == 3


def test_water_bottle_is_kept_with_three_distinct_frame_support():
    observations = {"water bottle": 6}
    frames = [
        {"objects": {"water bottle": 2}},
        {"objects": {"water bottle": 2}},
        {"objects": {"water bottle": 2}},
    ]

    SceneAnalyzer.filter_temporally_unsupported_objects(observations, frames)

    assert observations["water bottle"] == 6
