from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = Path(
    os.getenv(
        "SCENE_VIDEO_DIR",
        r"C:\Users\USER\Downloads\viiii",
    )
)
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
METADATA_CSV = LABEL_DIR / "video_metadata.csv"
SCENES_CSV = LABEL_DIR / "detected_scenes.csv"
ANNOTATIONS_CSV = LABEL_DIR / "annotations.csv"

SCENE_CATEGORIES = (
    "conversation",
    "romance",
    "action",
    "dance",
    "fight",
    "outdoor",
    "indoor",
    "travel",
    "crowd",
    "celebration",
    "dramatic",
    "other",
)

ANNOTATION_FIELDS = (
    "video",
    "start_time",
    "end_time",
    "scene_label",
    "scene_id",
)


def atomic_write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid video metadata: {video_path}")
        return {
            "filename": video_path.name,
            "duration_seconds": round(frame_count / fps, 3),
            "fps": round(fps, 3),
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
    finally:
        capture.release()


def frame_histogram(frame: np.ndarray) -> np.ndarray:
    small_frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )
    cv2.normalize(histogram, histogram)
    return histogram


def detect_scene_segments(
    video_path: Path,
    video_name: str | None = None,
    sample_interval_seconds: float = 1.0,
    cut_threshold: float = 0.58,
    min_scene_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than zero")
    if not 0 < cut_threshold <= 1:
        raise ValueError("cut_threshold must be between 0 and 1")
    if min_scene_seconds <= 0:
        raise ValueError("min_scene_seconds must be greater than zero")

    metadata = video_metadata(video_path)
    duration = metadata["duration_seconds"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    boundaries = [0.0]
    previous_histogram = None
    last_boundary = 0.0
    try:
        sample_time = 0.0
        while sample_time < duration:
            capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000)
            ok, frame = capture.read()
            if not ok:
                break
            current_histogram = frame_histogram(frame)
            if previous_histogram is not None:
                difference = float(cv2.compareHist(
                    previous_histogram,
                    current_histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                ))
                if (
                    difference >= cut_threshold
                    and sample_time - last_boundary >= min_scene_seconds
                    and duration - sample_time >= min_scene_seconds
                ):
                    boundaries.append(round(sample_time, 3))
                    last_boundary = sample_time
            previous_histogram = current_histogram
            sample_time += sample_interval_seconds
    finally:
        capture.release()

    boundaries.append(duration)
    source_name = video_name or video_path.name
    scenes = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        if end <= start:
            continue
        scenes.append({
            "scene_id": f"{source_name}::scene_{index:04d}",
            "video": source_name,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
        })
    return scenes


def capture_scene_preview(
    video_path: Path,
    start_time: float,
    end_time: float,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames = []
    try:
        for position in (0.1, 0.5, 0.9):
            timestamp = start_time + (end_time - start_time) * position
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
    finally:
        capture.release()
    return frames
