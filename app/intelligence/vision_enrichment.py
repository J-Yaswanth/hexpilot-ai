"""Optional OpenAI-compatible vision-language enrichment.

The module is deliberately independent of the local detectors.  It is a
no-op until an API key is supplied, so normal video analysis remains local.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request

from app.intelligence.scene_taxonomy import (
    SCENE_LABELS,
    SCENE_LABEL_GROUPS,
    SCENE_TAXONOMY,
)

logger = logging.getLogger(__name__)


class VisionProviderError(RuntimeError):
    """Raised when a configured vision provider cannot return valid data."""


@dataclass(frozen=True)
class VisionEnrichment:
    description: str
    category: str | None = None
    emotion: str | None = None
    narrative_type: str | None = None
    narrative_confidence: float | None = None
    narrative_evidence: list[str] = field(default_factory=list)
    dialogue_type: str | None = None
    dialogue_confidence: float | None = None
    dialogue_evidence: list[str] = field(default_factory=list)
    scene_labels: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    appearance: Any = field(default_factory=dict)
    objects: list[Any] = field(default_factory=list)


class VisionEnricher:
    """Call an OpenAI-compatible multimodal chat endpoint when configured."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_VISION_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )).rstrip("/")
        self.model = model or os.getenv("OPENAI_VISION_MODEL") or os.getenv(
            "OPENAI_MODEL", "gpt-4o-mini"
        )
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def enrich(self, frame: Any, scene_context: Mapping[str, Any] | str | None = None) -> VisionEnrichment | None:
        """Describe *frame* with optional scene context, or return ``None`` if disabled.

        ``frame`` may be image bytes, a path, a data URL, or a numpy/OpenCV frame.
        """
        if not self.enabled:
            logger.debug("Vision enrichment disabled: no API key configured")
            return None

        image_url = self._image_data_url(frame)
        context = scene_context if isinstance(scene_context, str) else json.dumps(
            scene_context or {}, ensure_ascii=False, default=str
        )
        taxonomy = json.dumps(
            SCENE_TAXONOMY,
            ensure_ascii=False,
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analyze this video frame. Return only JSON with keys: "
                        "description (string), category (string), actions (array of strings), "
                        "emotion (string or null), narrative_type (string or null), "
                        "narrative_confidence (number or null), "
                        "narrative_evidence (array of strings), appearance (object), "
                        "dialogue_type (string or null), "
                        "dialogue_confidence (number or null), "
                        "dialogue_evidence (array of strings), "
                        "scene_labels (array of objects with label, confidence "
                        "from 0 to 1, and evidence array), "
                        "objects (array of object observations). "
                        "For scene_labels, only use exact labels from this taxonomy: "
                        f"{taxonomy}. Return only labels supported by visible evidence "
                        "or the supplied timeline context; do not infer plot events, "
                        "emotions, or temporal cinematic effects from a single still. "
                        "Omit uncertain labels. "
                        f"Scene context: {context}"
                    )},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        req = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            logger.exception("Configured vision provider request failed")
            raise VisionProviderError("Vision provider request failed") from exc

        try:
            content = response_data["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict):
                raise ValueError("provider response is not an object")
            return VisionEnrichment(
                description=str(parsed.get("description", "")),
                category=parsed.get("category"),
                emotion=parsed.get("emotion"),
                narrative_type=parsed.get("narrative_type"),
                narrative_confidence=parsed.get("narrative_confidence"),
                narrative_evidence=list(
                    parsed.get("narrative_evidence", [])
                ),
                dialogue_type=parsed.get("dialogue_type"),
                dialogue_confidence=parsed.get("dialogue_confidence"),
                dialogue_evidence=list(
                    parsed.get("dialogue_evidence", [])
                ),
                scene_labels=self._parse_scene_labels(
                    parsed.get("scene_labels", [])
                ),
                actions=list(parsed.get("actions", [])),
                appearance=parsed.get("appearance", {}),
                objects=list(parsed.get("objects", parsed.get("object_observations", []))),
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.exception("Configured vision provider returned invalid JSON")
            raise VisionProviderError("Vision provider returned invalid JSON") from exc

    @staticmethod
    def _parse_scene_labels(raw_labels: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_labels, list):
            raise ValueError("scene_labels must be an array")

        labels = []
        seen = set()
        for raw_label in raw_labels:
            if not isinstance(raw_label, dict):
                logger.warning("Ignoring malformed scene label from vision provider")
                continue

            label_value = raw_label.get("label")
            if not isinstance(label_value, str):
                logger.warning("Ignoring scene label without a string label")
                continue
            label_key = label_value.strip().casefold()
            canonical_label = SCENE_LABELS.get(label_key)
            if canonical_label is None:
                logger.warning("Ignoring unsupported scene label: %s", label_value)
                continue
            if label_key in seen:
                continue

            confidence = raw_label.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                logger.warning(
                    "Ignoring scene label with invalid confidence: %s",
                    canonical_label,
                )
                continue

            evidence = raw_label.get("evidence", [])
            if (
                not isinstance(evidence, list)
                or not evidence
                or not all(isinstance(item, str) for item in evidence)
            ):
                logger.warning(
                    "Ignoring scene label with invalid evidence: %s",
                    canonical_label,
                )
                continue

            labels.append({
                "group": SCENE_LABEL_GROUPS[label_key][0],
                "groups": SCENE_LABEL_GROUPS[label_key],
                "label": canonical_label,
                "confidence": float(confidence),
                "evidence": evidence,
            })
            seen.add(label_key)

        return labels

    @staticmethod
    def _image_data_url(frame: Any) -> str:
        if isinstance(frame, str) and frame.startswith("data:image/"):
            return frame
        mime = "image/jpeg"
        if isinstance(frame, (str, os.PathLike)):
            path = Path(frame)
            mime = mimetypes.guess_type(path.name)[0] or mime
            data = path.read_bytes()
        elif isinstance(frame, (bytes, bytearray, memoryview)):
            data = bytes(frame)
        else:
            try:
                import cv2
                import numpy as np
                if not isinstance(frame, np.ndarray):
                    raise TypeError("frame must be image bytes, a path, or a numpy array")
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise ValueError("could not encode frame")
                data = encoded.tobytes()
            except ImportError as exc:
                raise TypeError("numpy/OpenCV is required for array frames") from exc
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
