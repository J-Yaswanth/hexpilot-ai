from __future__ import annotations

import random
from collections import Counter, defaultdict


def split_videos(
    records: list[dict],
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    attempts: int = 20_000,
    allow_incomplete_evaluation: bool = False,
) -> dict[str, list[str]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    label_videos: dict[str, set[str]] = defaultdict(set)
    videos = set()
    for record in records:
        video = str(record["video"])
        label = str(record["scene_label"])
        videos.add(video)
        label_videos[label].add(video)

    if len(videos) < 3:
        raise ValueError("At least three distinct source videos are required for grouped train/validation/test splits.")

    insufficient = {
        label: sorted(video_names)
        for label, video_names in label_videos.items()
        if len(video_names) < 3
    }
    if insufficient and not allow_incomplete_evaluation:
        details = "; ".join(
            f"{label}: {len(video_names)} video(s)"
            for label, video_names in sorted(insufficient.items())
        )
        raise ValueError(
            "Each scene label must occur in at least three distinct source videos "
            "so train, validation, and test can each contain that class without "
            f"video leakage. Insufficient coverage: {details}"
        )

    video_list = sorted(videos)
    train_count = max(1, round(len(video_list) * train_fraction))
    validation_count = max(1, round(len(video_list) * validation_fraction))
    if train_count + validation_count >= len(video_list):
        train_count = len(video_list) - 2
        validation_count = 1
    all_labels = set(label_videos)
    scene_counts = Counter(record["scene_label"] for record in records)
    target_train = train_fraction
    target_validation = validation_fraction
    best_split = None
    best_score = float("inf")
    rng = random.Random(seed)

    for _ in range(attempts):
        shuffled = video_list.copy()
        rng.shuffle(shuffled)
        split = {
            "train": shuffled[:train_count],
            "validation": shuffled[train_count:train_count + validation_count],
            "test": shuffled[train_count + validation_count:],
        }
        covered = {
            name: {
                record["scene_label"]
                for record in records
                if record["video"] in names
            }
            for name, names in split.items()
        }
        if allow_incomplete_evaluation and covered["train"] != all_labels:
            continue
        missing = sum(len(all_labels - covered[name]) for name in split)

        label_split_counts = {
            name: Counter(
                record["scene_label"]
                for record in records
                if record["video"] in names
            )
            for name, names in split.items()
        }
        distribution_error = 0.0
        for label, total_count in scene_counts.items():
            distribution_error += abs(
                label_split_counts["train"][label] / total_count - target_train
            )
            distribution_error += abs(
                label_split_counts["validation"][label] / total_count
                - target_validation
            )
        score = missing * 10_000 + distribution_error
        if score < best_score:
            best_score = score
            best_split = split
            if missing == 0 and distribution_error < 0.5:
                break

    if best_split is None:
        raise ValueError(
            "Could not create video-grouped splits with every class in training. "
            "Add labeled source videos for the missing classes."
        )
    split_labels = {
        name: {
            record["scene_label"]
            for record in records
            if record["video"] in names
        }
        for name, names in best_split.items()
    }
    uncovered = {
        name: sorted(all_labels - labels)
        for name, labels in split_labels.items()
        if all_labels - labels
        and (not allow_incomplete_evaluation or name == "train")
    }
    if uncovered:
        details = "; ".join(
            f"{name} missing {', '.join(labels)}"
            for name, labels in uncovered.items()
        )
        raise ValueError(
            "Could not place every class in every split using whole source videos. "
            f"Add more labeled source videos or combine rare classes. {details}"
        )
    return best_split


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_count: int,
) -> tuple[dict[str, float], list[list[int]]]:
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1

    per_class = []
    for index in range(class_count):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(class_count)) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))

    sample_count = len(y_true)
    metrics = {
        "accuracy": sum(matrix[index][index] for index in range(class_count)) / sample_count if sample_count else 0.0,
        "precision_macro": sum(item[0] for item in per_class) / class_count if class_count else 0.0,
        "recall_macro": sum(item[1] for item in per_class) / class_count if class_count else 0.0,
        "f1_macro": sum(item[2] for item in per_class) / class_count if class_count else 0.0,
    }
    return metrics, matrix
