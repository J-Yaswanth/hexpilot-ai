"""
AI Video Intelligence Studio
YOLO11 + DeepSORT Video Analysis Pipeline

This module:
1. Opens the uploaded video
2. Detects persons using YOLO11
3. Tracks persons using DeepSORT
4. Collects real frame-by-frame evidence
5. Creates an annotated output video
6. Returns structured analysis results

No hardcoded detection counts are used.
"""

from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

from app.vision.deep_sort_tracker import DeepSortPersonTracker


class VideoAnalysisPipeline:

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.50,
        output_dir="data/outputs",
    ):

        self.model_path = model_path
        self.confidence = confidence

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("INITIALIZING VIDEO ANALYSIS PIPELINE")
        print("=" * 70)

        print("\nLoading YOLO11...")
        self.model = YOLO(model_path)
        print("YOLO11 loaded.")

        print("\nLoading DeepSORT...")
        self.tracker = DeepSortPersonTracker()
        print("DeepSORT loaded.")

    def analyze(self, video_path, output_name="processed_video.mp4"):

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video does not exist: {video_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if fps <= 0:
            fps = 25.0

        duration = (
            total_frames / fps
            if fps > 0
            else 0
        )

        print("\nVIDEO INFORMATION")
        print("-" * 70)
        print(f"Resolution : {width} x {height}")
        print(f"FPS        : {fps:.2f}")
        print(f"Frames     : {total_frames}")
        print(f"Duration   : {duration:.2f} sec")

        output_path = (
            self.output_dir / output_name
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                f"Could not create output video: "
                f"{output_path}"
            )

        # ------------------------------------------------------------
        # REAL ANALYSIS DATA
        # ------------------------------------------------------------

        frame_results = []

        unique_track_ids = set()

        track_history = defaultdict(
            lambda: {
                "frames": [],
                "confidences": [],
                "positions": [],
            }
        )

        max_simultaneous_people = 0

        processed_frames = 0

        print("\n")
        print("=" * 70)
        print("STARTING YOLO11 + DEEPSORT ANALYSIS")
        print("=" * 70)

        # ------------------------------------------------------------
        # PROCESS VIDEO
        # ------------------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            processed_frames += 1

            # --------------------------------------------------------
            # YOLO11 PERSON DETECTION
            # --------------------------------------------------------

            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                classes=[0],          # COCO class 0 = person
                verbose=False,
            )

            detections = []

            if (
                results
                and results[0].boxes is not None
            ):

                boxes = results[0].boxes

                for box in boxes:

                    xyxy = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        xyxy
                    )

                    confidence = float(
                        box.conf[0]
                        .cpu()
                        .item()
                    )

                    width_box = x2 - x1
                    height_box = y2 - y1

                    if (
                        width_box <= 0
                        or height_box <= 0
                    ):
                        continue

                    detections.append(
                        {
                            "bbox": [
                                x1,
                                y1,
                                x2,
                                y2,
                            ],
                            "confidence": confidence,
                            "class_name": "person",
                        }
                    )

            # --------------------------------------------------------
            # DEEPSORT TRACKING
            # --------------------------------------------------------

            tracks = self.tracker.update(
                detections,
                frame,
            )

            current_track_ids = []

            # --------------------------------------------------------
            # PROCESS TRACKS
            # --------------------------------------------------------

            for track in tracks:

                track_id = str(
                    track["track_id"]
                )

                bbox = track["bbox"]

                confidence = float(
                    track.get(
                        "confidence",
                        0.0
                    )
                )

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                # ----------------------------------------------------
                # PERSON CENTER
                # ----------------------------------------------------

                center_x = (
                    x1 + x2
                ) // 2

                center_y = (
                    y1 + y2
                ) // 2

                current_track_ids.append(
                    track_id
                )

                unique_track_ids.add(
                    track_id
                )

                # ----------------------------------------------------
                # STORE FRAME HISTORY
                # ----------------------------------------------------

                track_history[
                    track_id
                ]["frames"].append(
                    processed_frames
                )

                # ----------------------------------------------------
                # STORE CONFIDENCE
                # ----------------------------------------------------

                track_history[
                    track_id
                ]["confidences"].append(
                    confidence
                )

                # ----------------------------------------------------
                # STORE POSITION
                # ----------------------------------------------------

                track_history[
                    track_id
                ]["positions"].append(
                    {
                        "frame": processed_frames,
                        "x": center_x,
                        "y": center_y,
                    }
                )

                # ----------------------------------------------------
                # DRAW TRACKING BOX
                # ----------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"Person ID {track_id} "
                    f"| {confidence:.2f}"
                )

                cv2.rectangle(
                    frame,
                    (
                        x1,
                        max(0, y1 - 28)
                    ),
                    (
                        x1 + 190,
                        y1,
                    ),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1 + 5,
                        max(18, y1 - 8)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                )

            # --------------------------------------------------------
            # CURRENT PEOPLE
            # --------------------------------------------------------

            current_people = len(
                current_track_ids
            )

            max_simultaneous_people = max(
                max_simultaneous_people,
                current_people,
            )

            # --------------------------------------------------------
            # SAVE FRAME EVIDENCE
            # --------------------------------------------------------

            frame_results.append(
                {
                    "frame": processed_frames,
                    "timestamp": (
                        processed_frames / fps
                    ),
                    "person_detections": len(
                        detections
                    ),
                    "active_track_ids":
                        current_track_ids,
                    "active_people":
                        current_people,
                }
            )

            # --------------------------------------------------------
            # DISPLAY INFO ON VIDEO
            # --------------------------------------------------------

            cv2.putText(
                frame,
                f"Frame: "
                f"{processed_frames}/"
                f"{total_frames}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"People: {current_people}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

            # --------------------------------------------------------
            # TERMINAL PROGRESS
            # --------------------------------------------------------

            if processed_frames % 25 == 0:

                print(
                    f"Frame "
                    f"{processed_frames:4d} | "
                    f"Detections: "
                    f"{len(detections):2d} | "
                    f"Active people: "
                    f"{current_people:2d} | "
                    f"IDs: "
                    f"{current_track_ids}"
                )

        # ------------------------------------------------------------
        # RELEASE VIDEO
        # ------------------------------------------------------------

        cap.release()
        writer.release()

        # ------------------------------------------------------------
        # BUILD TRACK SUMMARY
        # ------------------------------------------------------------

        track_summary = []

        for track_id in sorted(
            unique_track_ids,
            key=lambda x:
                int(x)
                if x.isdigit()
                else x,
        ):

            frames = (
                track_history[
                    track_id
                ]["frames"]
            )

            confidences = (
                track_history[
                    track_id
                ]["confidences"]
            )

            positions = (
                track_history[
                    track_id
                ]["positions"]
            )

            if not frames:
                continue

            # --------------------------------------------------------
            # AVERAGE CONFIDENCE
            # --------------------------------------------------------

            average_confidence = (
                sum(confidences)
                / len(confidences)
                if confidences
                else 0.0
            )

            # --------------------------------------------------------
            # TOTAL MOVEMENT
            # --------------------------------------------------------

            total_movement = 0.0

            for i in range(
                1,
                len(positions)
            ):

                previous = (
                    positions[i - 1]
                )

                current = (
                    positions[i]
                )

                dx = (
                    current["x"]
                    - previous["x"]
                )

                dy = (
                    current["y"]
                    - previous["y"]
                )

                distance = (
                    dx ** 2
                    + dy ** 2
                ) ** 0.5

                total_movement += (
                    distance
                )

            # --------------------------------------------------------
            # TRACK SUMMARY
            # --------------------------------------------------------

            track_summary.append(
                {
                    "track_id": track_id,
                    "frames_tracked": len(
                        frames
                    ),
                    "first_frame": min(
                        frames
                    ),
                    "last_frame": max(
                        frames
                    ),
                    "average_confidence":
                        average_confidence,
                    "total_movement_pixels":
                        round(
                            total_movement,
                            2
                        ),
                }
            )

        # ------------------------------------------------------------
        # FINAL RESULTS
        # ------------------------------------------------------------

        results = {

            "video": {

                "path": str(
                    video_path
                ),

                "width": width,

                "height": height,

                "fps": fps,

                "total_frames":
                    total_frames,

                "duration":
                    duration,
            },

            "analysis": {

                "frames_processed":
                    processed_frames,

                "max_simultaneous_people":
                    max_simultaneous_people,

                "total_unique_track_ids":
                    len(unique_track_ids),
            },

            "frame_results":
                frame_results,

            "tracks":
                track_summary,

            "output_video":
                str(output_path),
        }

        # ------------------------------------------------------------
        # FINAL TERMINAL OUTPUT
        # ------------------------------------------------------------

        print("\n")
        print("=" * 70)
        print("VIDEO ANALYSIS COMPLETED")
        print("=" * 70)

        print(
            f"Frames processed          : "
            f"{processed_frames}"
        )

        print(
            f"Maximum simultaneous people : "
            f"{max_simultaneous_people}"
        )

        print(
            f"Unique DeepSORT identities : "
            f"{len(unique_track_ids)}"
        )

        print(
            f"Output video              : "
            f"{output_path}"
        )

        print("=" * 70)

        # ------------------------------------------------------------
        # TRACK DETAILS
        # ------------------------------------------------------------

        print("\n")
        print("TRACK DETAILS")
        print("-" * 70)

        for track in track_summary:

            print(
                f"Person #{track['track_id']} | "
                f"Frames: "
                f"{track['frames_tracked']} | "
                f"First: "
                f"{track['first_frame']} | "
                f"Last: "
                f"{track['last_frame']} | "
                f"Avg confidence: "
                f"{track['average_confidence']:.2f} | "
                f"Movement: "
                f"{track['total_movement_pixels']:.1f}px"
            )

        return results