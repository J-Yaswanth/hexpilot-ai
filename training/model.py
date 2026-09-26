from __future__ import annotations

from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

def make_model(class_count: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    return nn.Sequential(
        backbone,
        nn.Linear(512, class_count),
    )


def segment_tensor(
    video_path: Path,
    start_time: float,
    end_time: float,
    device: torch.device,
) -> torch.Tensor:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        timestamps = (
            start_time + (end_time - start_time) * fraction
            for fraction in (0.15, 0.5, 0.85)
        )
        frames = []
        transform = ResNet18_Weights.DEFAULT.transforms()
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image

            frames.append(transform(Image.fromarray(rgb)))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(
            f"No frames could be read from segment {start_time:.3f}–{end_time:.3f} "
            f"in {video_path}"
        )
    return torch.stack(frames).to(device)


@torch.inference_mode()
def extract_segment_features(
    model: nn.Sequential,
    frames: torch.Tensor,
) -> torch.Tensor:
    backbone = model[0]
    features = backbone(frames)
    return features.mean(dim=0)


@torch.inference_mode()
def predict_segment(
    model: nn.Sequential,
    video_path: Path,
    start_time: float,
    end_time: float,
    device: torch.device,
) -> tuple[int, float]:
    frames = segment_tensor(video_path, start_time, end_time, device)
    logits = model(frames).mean(dim=0)
    probabilities = torch.softmax(logits, dim=0)
    confidence, label_index = probabilities.max(dim=0)
    return int(label_index.item()), float(confidence.item())
