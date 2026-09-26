from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from training.common import (
    ANNOTATIONS_CSV,
    DEFAULT_VIDEO_DIR,
    PROJECT_ROOT,
    SCENE_CATEGORIES,
    atomic_write_csv,
    read_csv_rows,
)
from training.dataset import classification_metrics, split_videos
from training.model import extract_segment_features, make_model, segment_tensor

MODEL_DIR = PROJECT_ROOT / "models" / "scene_classifier"
MIN_SEGMENTS_PER_CLASS = 6


def read_labeled_records(
    annotations_path: Path,
    video_dir: Path,
    minimum_segments_per_class: int = MIN_SEGMENTS_PER_CLASS,
) -> list[dict]:
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Annotations CSV not found: {annotations_path}")
    rows = read_csv_rows(annotations_path)
    valid_labels = {label.casefold(): label for label in SCENE_CATEGORIES}
    records = []
    for line_number, row in enumerate(rows, start=2):
        label_value = row.get("scene_label", "").strip()
        if not label_value:
            continue
        canonical_label = valid_labels.get(label_value.casefold())
        if canonical_label is None:
            raise ValueError(
                f"Unsupported scene label on CSV line {line_number}: {label_value}"
            )
        video_name = row.get("video", "").strip()
        if not video_name or Path(video_name).name != video_name:
            raise ValueError(
                f"Invalid video filename on CSV line {line_number}: {video_name}"
            )
        video_path = video_dir / video_name
        if not video_path.is_file():
            raise FileNotFoundError(f"Video for annotation not found: {video_path}")
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        if start_time < 0 or end_time <= start_time:
            raise ValueError(f"Invalid time range on CSV line {line_number}")
        records.append({
            "video": video_name,
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "scene_label": canonical_label,
            "scene_id": row.get("scene_id", ""),
        })
    if not records:
        raise ValueError(
            "No confirmed labels exist yet. Label scenes in the labeling tool before training."
        )
    counts = Counter(record["scene_label"] for record in records)
    insufficient = {
        label: count
        for label, count in counts.items()
        if count < minimum_segments_per_class
    }
    if insufficient:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(insufficient.items())
        )
        raise ValueError(
            f"Not enough labeled scenes for training: each present category needs at "
            f"least {minimum_segments_per_class} labeled segments. Current counts: {details}"
        )
    return records


def build_features(
    records: list[dict],
    model: nn.Sequential,
    device: torch.device,
) -> torch.Tensor:
    features = []
    for index, record in enumerate(records, start=1):
        print(
            f"Extracting pretrained visual features "
            f"{index}/{len(records)}: {record['video']} "
            f"{record['start_time']:.2f}–{record['end_time']:.2f}s"
        )
        frames = segment_tensor(
            record["video_path"],
            record["start_time"],
            record["end_time"],
            device,
        )
        feature = extract_segment_features(model, frames)
        features.append(feature.cpu())
    return torch.stack(features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 transfer-learning classifier with video-level splits."
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experimental",
        action="store_true",
        help=(
            "Allow small classes and incomplete validation/test class coverage. "
            "Source videos still remain disjoint, and every class must be in training."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = read_labeled_records(
        args.annotations,
        args.video_dir,
        minimum_segments_per_class=0 if args.experimental else MIN_SEGMENTS_PER_CLASS,
    )
    split = split_videos(
        records,
        seed=args.seed,
        allow_incomplete_evaluation=args.experimental,
    )
    print(
        f"Confirmed labeled scenes: {len(records)} | "
        f"source videos: {len({record['video'] for record in records})}"
    )
    print(
        "Video-level split: "
        + json.dumps({key: len(value) for key, value in split.items()})
    )
    if args.experimental:
        print(
            "EXPERIMENTAL MODE: source videos remain disjoint, but validation/test "
            "may not contain every class; evaluation is not submission-grade."
        )

    classes = sorted({record["scene_label"] for record in records})
    model = make_model(len(classes), pretrained=True).to(device)
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    model[0].eval()
    features = build_features(records, model, device)
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = torch.tensor(
        [class_to_index[record["scene_label"]] for record in records],
        dtype=torch.long,
    )
    indices_by_split = {
        name: [
            index for index, record in enumerate(records)
            if record["video"] in videos
        ]
        for name, videos in split.items()
    }

    head = nn.Linear(features.shape[1], len(classes))
    train_indices = indices_by_split["train"]
    train_counts = Counter(labels[index].item() for index in train_indices)
    class_weights = torch.tensor([
        len(train_indices) / (len(classes) * max(train_counts.get(index, 0), 1))
        for index in range(len(classes))
    ])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=1e-3)
    best_validation_f1 = -1.0
    best_head_state = None
    patience = 15
    stale_epochs = 0

    for epoch in range(args.epochs):
        head.train()
        optimizer.zero_grad()
        train_logits = head(features[train_indices])
        loss = criterion(train_logits, labels[train_indices])
        loss.backward()
        optimizer.step()

        head.eval()
        validation_indices = indices_by_split["validation"]
        with torch.inference_mode():
            validation_predictions = head(features[validation_indices]).argmax(dim=1)
        validation_metrics, _ = classification_metrics(
            labels[validation_indices].tolist(),
            validation_predictions.tolist(),
            len(classes),
        )
        validation_f1 = validation_metrics["f1_macro"]
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_head_state = {
                key: value.detach().clone()
                for key, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={loss.item():.4f} val_macro_f1={validation_f1:.4f}"
            )
        if stale_epochs >= patience:
            print(f"Early stopping after epoch {epoch + 1}.")
            break

    if best_head_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")
    head.load_state_dict(best_head_state)
    model[1].load_state_dict(best_head_state)
    model.eval()

    test_indices = indices_by_split["test"]
    with torch.inference_mode():
        test_logits = head(features[test_indices])
        test_predictions = test_logits.argmax(dim=1).tolist()
    test_truth = labels[test_indices].tolist()
    metrics, confusion = classification_metrics(
        test_truth,
        test_predictions,
        len(classes),
    )
    class_metrics = {}
    for class_index, label in enumerate(classes):
        true_positive = confusion[class_index][class_index]
        false_positive = sum(row[class_index] for row in confusion) - true_positive
        false_negative = sum(confusion[class_index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[class_index]),
        }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "classes": classes,
        "architecture": "resnet18_frozen_backbone_linear_head",
        "input_size": 224,
        "seed": args.seed,
        "split_video_names": split,
    }, args.model_dir / "scene_classifier.pt")
    with (args.model_dir / "video_split.json").open("w", encoding="utf-8") as output:
        json.dump(split, output, indent=2)
    evaluation = {
        **metrics,
        "class_metrics": class_metrics,
        "test_scene_count": len(test_indices),
        "labeled_scene_count": len(records),
        "class_names": classes,
        "validation_best_macro_f1": best_validation_f1,
        "experimental_mode": args.experimental,
        "test_class_counts": {
            label: sum(
                record["scene_label"] == label
                for record_index, record in enumerate(records)
                if record_index in test_indices
            )
            for label in classes
        },
        "note": (
            "Experimental held-out source-video evaluation; class coverage in "
            "validation/test may be incomplete."
            if args.experimental
            else "Evaluation uses a held-out source-video test split."
        ),
    }
    with (args.model_dir / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(evaluation, output, indent=2)
    matrix_rows = [
        {"actual_label": label, **{
            f"predicted_{predicted_label}": confusion[row_index][column_index]
            for column_index, predicted_label in enumerate(classes)
        }}
        for row_index, label in enumerate(classes)
    ]
    atomic_write_csv(
        args.model_dir / "confusion_matrix.csv",
        ("actual_label", *(f"predicted_{label}" for label in classes)),
        matrix_rows,
    )
    prediction_rows = []
    for record_index, predicted_index in zip(test_indices, test_predictions):
        record = records[record_index]
        prediction_rows.append({
            "video": record["video"],
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "actual_label": record["scene_label"],
            "predicted_label": classes[predicted_index],
        })
    atomic_write_csv(
        args.model_dir / "test_predictions.csv",
        ("video", "start_time", "end_time", "actual_label", "predicted_label"),
        prediction_rows,
    )
    print("\nHeld-out test evaluation:")
    print(json.dumps(evaluation, indent=2))
    print(f"Saved model and evaluation artifacts in {args.model_dir}")


if __name__ == "__main__":
    main()
