import pytest

from training.common import ANNOTATION_FIELDS, atomic_write_csv, read_csv_rows
from training.dataset import classification_metrics, split_videos
from training.prepare_dataset import prepare_dataset


def test_group_split_keeps_source_videos_disjoint_and_classes_in_each_split():
    records = []
    for video_index in range(12):
        for label in ("conversation", "action"):
            records.append({
                "video": f"video_{video_index}.mp4",
                "scene_label": label,
            })

    split = split_videos(records, seed=8)

    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert set().union(*map(set, split.values())) == {
        f"video_{video_index}.mp4" for video_index in range(12)
    }
    for video_names in split.values():
        assert {
            record["scene_label"]
            for record in records
            if record["video"] in video_names
        } == {"conversation", "action"}


def test_group_split_refuses_label_with_inadequate_video_coverage():
    records = [
        {"video": "one.mp4", "scene_label": "rare"},
        {"video": "two.mp4", "scene_label": "rare"},
        {"video": "three.mp4", "scene_label": "common"},
        {"video": "four.mp4", "scene_label": "common"},
        {"video": "five.mp4", "scene_label": "common"},
    ]

    with pytest.raises(ValueError, match="three distinct source videos"):
        split_videos(records)


def test_experimental_group_split_keeps_rare_class_in_training():
    records = [
        {"video": "rare.mp4", "scene_label": "rare"},
        *[
            {"video": f"common_{index}.mp4", "scene_label": "common"}
            for index in range(5)
        ],
    ]

    split = split_videos(records, seed=8, allow_incomplete_evaluation=True)

    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    assert "rare.mp4" in split["train"]
    assert {
        record["scene_label"]
        for record in records
        if record["video"] in split["train"]
    } == {"rare", "common"}


def test_classification_metrics_and_confusion_matrix():
    metrics, confusion = classification_metrics(
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        class_count=2,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision_macro"] == pytest.approx(0.8333333333)
    assert metrics["recall_macro"] == pytest.approx(0.75)
    assert confusion == [[2, 0], [1, 1]]


def test_prepare_dataset_leaves_labels_blank_and_preserves_confirmed_labels(
    tmp_path,
    monkeypatch,
):
    import training.prepare_dataset as preparation

    video_dir = tmp_path / "videos"
    label_dir = tmp_path / "labels"
    video_dir.mkdir()
    (video_dir / "sample.mp4").touch()
    monkeypatch.setattr(
        preparation,
        "video_metadata",
        lambda path: {
            "filename": path.name,
            "duration_seconds": 4.0,
            "fps": 24.0,
            "resolution": "640x360",
            "width": 640,
            "height": 360,
            "frame_count": 96,
        },
    )
    monkeypatch.setattr(
        preparation,
        "detect_scene_segments",
        lambda path, **kwargs: [{
            "scene_id": "sample.mp4::scene_0001",
            "video": path.name,
            "start_time": 0.0,
            "end_time": 4.0,
        }],
    )

    prepare_dataset(video_dir, label_dir)
    annotations_path = label_dir / "annotations.csv"
    annotations = read_csv_rows(annotations_path)
    assert annotations[0]["scene_label"] == ""
    assert set(annotations[0]) >= set(ANNOTATION_FIELDS)

    annotations[0]["scene_label"] = "conversation"
    atomic_write_csv(annotations_path, ANNOTATION_FIELDS, annotations)
    prepare_dataset(video_dir, label_dir)

    assert read_csv_rows(annotations_path)[0]["scene_label"] == "conversation"
