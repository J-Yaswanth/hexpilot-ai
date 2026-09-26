from __future__ import annotations

import argparse
from pathlib import Path

from training.common import (
    ANNOTATIONS_CSV,
    ANNOTATION_FIELDS,
    DEFAULT_VIDEO_DIR,
    LABEL_DIR,
    METADATA_CSV,
    SCENES_CSV,
    atomic_write_csv,
    detect_scene_segments,
    read_csv_rows,
    video_metadata,
)


def prepare_dataset(
    video_dir: Path,
    label_dir: Path = LABEL_DIR,
    sample_interval_seconds: float = 1.0,
    cut_threshold: float = 0.58,
    min_scene_seconds: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    video_paths = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    )
    if not video_paths:
        raise FileNotFoundError(f"No supported video files found in: {video_dir}")

    metadata_rows = []
    scene_rows = []
    for index, video_path in enumerate(video_paths, start=1):
        print(f"[{index}/{len(video_paths)}] Inspecting {video_path.name}")
        metadata_rows.append(video_metadata(video_path))
        scene_rows.extend(detect_scene_segments(
            video_path,
            sample_interval_seconds=sample_interval_seconds,
            cut_threshold=cut_threshold,
            min_scene_seconds=min_scene_seconds,
        ))

    label_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = label_dir / METADATA_CSV.name
    scenes_path = label_dir / SCENES_CSV.name
    annotations_path = label_dir / ANNOTATIONS_CSV.name

    atomic_write_csv(
        metadata_path,
        (
            "filename",
            "duration_seconds",
            "fps",
            "resolution",
            "width",
            "height",
            "frame_count",
        ),
        metadata_rows,
    )
    atomic_write_csv(
        scenes_path,
        ("scene_id", "video", "start_time", "end_time"),
        scene_rows,
    )

    prior_rows = read_csv_rows(annotations_path)
    def time_key(value: str) -> float | None:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    prior_labels = {
        (
            row.get("video", ""),
            time_key(row.get("start_time", "")),
            time_key(row.get("end_time", "")),
        ):
        row.get("scene_label", "")
        for row in prior_rows
    }
    annotation_rows = []
    for scene in scene_rows:
        match = (
            scene["video"],
            round(scene["start_time"], 3),
            round(scene["end_time"], 3),
        )
        annotation_rows.append({
            **scene,
            "scene_label": prior_labels.get(match, ""),
        })
    atomic_write_csv(annotations_path, ANNOTATION_FIELDS, annotation_rows)

    print(f"\nVideos inspected: {len(metadata_rows)}")
    print(f"Candidate scenes: {len(scene_rows)}")
    print(f"Metadata: {metadata_path}")
    print(f"Scene segments: {scenes_path}")
    print(f"Annotations (labels left blank): {annotations_path}")
    return metadata_rows, scene_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect videos and create candidate scene-segment CSVs."
    )
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--cut-threshold", type=float, default=0.58)
    parser.add_argument("--min-scene-seconds", type=float, default=2.0)
    args = parser.parse_args()

    prepare_dataset(
        args.video_dir,
        args.label_dir,
        args.sample_interval,
        args.cut_threshold,
        args.min_scene_seconds,
    )


if __name__ == "__main__":
    main()
