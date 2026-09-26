from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math


@dataclass
class PersonTrack:
    track_id: int
    bbox: List[int]
    confidence: float

    first_frame: int
    last_frame: int

    hits: int = 1
    missed_frames: int = 0

    confirmed: bool = False


class PersonIdentityTracker:
    """
    Lightweight person identity tracker.

    This tracker is intentionally separate from YOLO.

    YOLO answers:
        "Where are people in this frame?"

    This module answers:
        "Which detection most likely belongs to
         an already existing person?"

    It uses bounding-box center distance and IoU.
    """

    def __init__(
        self,
        max_missed_frames: int = 20,
        min_hits: int = 3,
        max_center_distance: float = 300.0,
        min_iou: float = 0.05,
    ):

        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.max_center_distance = max_center_distance
        self.min_iou = min_iou

        self.next_track_id = 1

        self.tracks: Dict[int, PersonTrack] = {}

        self.frame_number = 0

    # =========================================================
    # BASIC GEOMETRY
    # =========================================================

    @staticmethod
    def center(
        bbox: List[int],
    ) -> Tuple[float, float]:

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def iou(
        box_a: List[int],
        box_b: List[int],
    ) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(
            ax1,
            bx1,
        )

        intersection_y1 = max(
            ay1,
            by1,
        )

        intersection_x2 = min(
            ax2,
            bx2,
        )

        intersection_y2 = min(
            ay2,
            by2,
        )

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            ax2 - ax1,
        ) * max(
            0,
            ay2 - ay1,
        )

        area_b = max(
            0,
            bx2 - bx1,
        ) * max(
            0,
            by2 - by1,
        )

        union_area = (
            area_a
            + area_b
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    @classmethod
    def center_distance(
        cls,
        bbox_a: List[int],
        bbox_b: List[int],
    ) -> float:

        ax, ay = cls.center(
            bbox_a
        )

        bx, by = cls.center(
            bbox_b
        )

        return math.sqrt(
            (ax - bx) ** 2
            +
            (ay - by) ** 2
        )

    # =========================================================
    # MATCH SCORE
    # =========================================================

    def match_score(
        self,
        track_bbox: List[int],
        detection_bbox: List[int],
    ) -> Optional[float]:

        distance = self.center_distance(
            track_bbox,
            detection_bbox,
        )

        overlap = self.iou(
            track_bbox,
            detection_bbox,
        )

        # A detection is considered compatible if:
        #
        # 1. It is spatially close enough, OR
        # 2. Its bounding box overlaps the previous one.

        if (
            distance > self.max_center_distance
            and overlap < self.min_iou
        ):
            return None

        # Normalize distance.
        distance_score = max(
            0.0,
            1.0
            -
            (
                distance
                /
                self.max_center_distance
            )
        )

        # Combine spatial proximity and IoU.
        score = (
            0.65 * distance_score
            +
            0.35 * overlap
        )

        return score

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        detections: List[dict],
    ) -> List[dict]:

        self.frame_number += 1

        # -----------------------------------------------------
        # Only person detections
        # -----------------------------------------------------

        person_detections = []

        for detection in detections:

            if (
                detection.get(
                    "label",
                    ""
                ).lower()
                != "person"
            ):
                continue

            bbox = detection.get(
                "bbox"
            )

            confidence = float(
                detection.get(
                    "confidence",
                    0.0
                )
            )

            if (
                bbox is None
                or len(bbox) != 4
            ):
                continue

            person_detections.append(
                {
                    "bbox": [
                        int(v)
                        for v in bbox
                    ],
                    "confidence":
                        confidence,
                }
            )

        # -----------------------------------------------------
        # Existing tracks become missed temporarily.
        # -----------------------------------------------------

        for track in self.tracks.values():

            track.missed_frames += 1

        # -----------------------------------------------------
        # Build possible matches.
        # -----------------------------------------------------

        candidates = []

        for detection_index, detection in enumerate(
            person_detections
        ):

            for track_id, track in self.tracks.items():

                score = self.match_score(
                    track.bbox,
                    detection["bbox"],
                )

                if score is None:
                    continue

                candidates.append(
                    (
                        score,
                        track_id,
                        detection_index,
                    )
                )

        # Best matches first.
        candidates.sort(
            reverse=True
        )

        matched_tracks = set()
        matched_detections = set()

        current_results = []

        # -----------------------------------------------------
        # Apply matches
        # -----------------------------------------------------

        for (
            score,
            track_id,
            detection_index,
        ) in candidates:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            track = self.tracks[
                track_id
            ]

            detection = person_detections[
                detection_index
            ]

            track.bbox = detection[
                "bbox"
            ]

            track.confidence = detection[
                "confidence"
            ]

            track.last_frame = (
                self.frame_number
            )

            track.hits += 1

            track.missed_frames = 0

            if (
                track.hits
                >= self.min_hits
            ):
                track.confirmed = True

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                detection_index
            )

            current_results.append(
                {
                    "track_id":
                        track_id,

                    "bbox":
                        track.bbox,

                    "confidence":
                        track.confidence,

                    "confirmed":
                        track.confirmed,
                }
            )

        # -----------------------------------------------------
        # Create new tracks
        # -----------------------------------------------------

        for detection_index, detection in enumerate(
            person_detections
        ):

            if detection_index in matched_detections:
                continue

            track_id = (
                self.next_track_id
            )

            self.next_track_id += 1

            track = PersonTrack(
                track_id=track_id,

                bbox=detection[
                    "bbox"
                ],

                confidence=detection[
                    "confidence"
                ],

                first_frame=self.frame_number,

                last_frame=self.frame_number,

                hits=1,

                missed_frames=0,

                confirmed=(
                    self.min_hits <= 1
                ),
            )

            self.tracks[
                track_id
            ] = track

            current_results.append(
                {
                    "track_id":
                        track_id,

                    "bbox":
                        track.bbox,

                    "confidence":
                        track.confidence,

                    "confirmed":
                        track.confirmed,
                }
            )

        # -----------------------------------------------------
        # Remove tracks that disappeared for too long.
        # -----------------------------------------------------

        expired_ids = []

        for track_id, track in self.tracks.items():

            if (
                track.missed_frames
                >
                self.max_missed_frames
            ):
                expired_ids.append(
                    track_id
                )

        for track_id in expired_ids:

            del self.tracks[
                track_id
            ]

        return current_results

    # =========================================================
    # CONFIRMED TRACKS
    # =========================================================

    def get_confirmed_tracks(self):

        return [
            track
            for track in self.tracks.values()
            if track.confirmed
        ]

    # =========================================================
    # TRACK COUNT
    # =========================================================

    def get_confirmed_count(self):

        return len(
            self.get_confirmed_tracks()
        )

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_summary(self):

        confirmed = (
            self.get_confirmed_tracks()
        )

        return {
            "active_confirmed_tracks":
                len(confirmed),

            "track_ids":
                sorted(
                    track.track_id
                    for track in confirmed
                ),

            "tracks": [
                {
                    "track_id":
                        track.track_id,

                    "hits":
                        track.hits,

                    "confidence":
                        round(
                            track.confidence,
                            3
                        ),

                    "first_frame":
                        track.first_frame,

                    "last_frame":
                        track.last_frame,

                    "missed_frames":
                        track.missed_frames,
                }

                for track in confirmed
            ],
        }