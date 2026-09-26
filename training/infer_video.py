from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.common import DEFAULT_VIDEO_DIR, detect_scene_segments
from training.model import make_model, predict_segment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect scene cuts and classify each scene in a new video."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "scene_classifier"
        / "scene_classifier.pt",
    )
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--cut-threshold", type=float, default=0.58)
    parser.add_argument("--min-scene-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    if not args.model.is_file():
        parser.error(
            f"Trained model not found: {args.model}. "
            "Complete annotation and training first."
        )

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = make_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    scenes = detect_scene_segments(
        args.video,
        sample_interval_seconds=args.sample_interval,
        cut_threshold=args.cut_threshold,
        min_scene_seconds=args.min_scene_seconds,
    )
    results = []
    for scene in scenes:
        predicted_index, confidence = predict_segment(
            model,
            args.video,
            scene["start_time"],
            scene["end_time"],
            device,
        )
        results.append({
            "scene_start": scene["start_time"],
            "scene_end": scene["end_time"],
            "predicted_label": classes[predicted_index],
            "confidence": round(confidence, 4),
        })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
