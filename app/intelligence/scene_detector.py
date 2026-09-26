import cv2
import os


class SceneDetector:

    def __init__(
        self,
        threshold=0.35,
        min_scene_duration=1.0,
    ):
        """
        Scene change detector.

        threshold:
            Controls how much visual change is required
            before a new scene is detected.

        min_scene_duration:
            Minimum duration between scene changes.
        """

        self.threshold = threshold
        self.min_scene_duration = min_scene_duration

    # ---------------------------------------------------------
    # FRAME DIFFERENCE
    # ---------------------------------------------------------

    def calculate_frame_difference(
        self,
        previous_frame,
        current_frame,
    ):
        """
        Calculate visual difference between two frames.

        Returns a value between 0 and 1.
        """

        previous_gray = cv2.cvtColor(
            previous_frame,
            cv2.COLOR_BGR2GRAY,
        )

        current_gray = cv2.cvtColor(
            current_frame,
            cv2.COLOR_BGR2GRAY,
        )

        previous_gray = cv2.resize(
            previous_gray,
            (320, 180),
        )

        current_gray = cv2.resize(
            current_gray,
            (320, 180),
        )

        difference = cv2.absdiff(
            previous_gray,
            current_gray,
        )

        difference_score = (
            difference.mean() / 255.0
        )

        return float(difference_score)

    # ---------------------------------------------------------
    # DETECT SCENES
    # ---------------------------------------------------------

    def detect_scenes(self, video_path):

        if not os.path.exists(video_path):

            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:

            fps = 25.0

        duration = (
            total_frames / fps
        )

        print("=" * 70)
        print("SCENE DETECTION")
        print("=" * 70)

        print(
            f"Video        : {video_path}"
        )

        print(
            f"FPS          : {fps:.2f}"
        )

        print(
            f"Total frames : {total_frames}"
        )

        print(
            f"Duration     : {duration:.2f} sec"
        )

        print("=" * 70)

        # -----------------------------------------------------
        # VARIABLES
        # -----------------------------------------------------

        scenes = []

        scene_number = 1

        scene_start_frame = 1

        previous_frame = None

        frame_number = 0

        last_scene_change_time = 0.0

        # -----------------------------------------------------
        # READ VIDEO
        # -----------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # First frame
            if previous_frame is None:

                previous_frame = frame

                continue

            current_time = (
                frame_number / fps
            )

            # -------------------------------------------------
            # CALCULATE VISUAL DIFFERENCE
            # -------------------------------------------------

            difference = (
                self.calculate_frame_difference(
                    previous_frame,
                    frame,
                )
            )

            # -------------------------------------------------
            # SCENE CHANGE
            # -------------------------------------------------

            enough_time_passed = (
                current_time
                - last_scene_change_time
                >= self.min_scene_duration
            )

            if (
                difference >= self.threshold
                and enough_time_passed
            ):

                scene_end_frame = (
                    frame_number - 1
                )

                scene_start_time = (
                    scene_start_frame / fps
                )

                scene_end_time = (
                    scene_end_frame / fps
                )

                scenes.append(
                    {
                        "scene_id": scene_number,
                        "start_frame": scene_start_frame,
                        "end_frame": scene_end_frame,
                        "start_time": round(
                            scene_start_time,
                            2,
                        ),
                        "end_time": round(
                            scene_end_time,
                            2,
                        ),
                        "duration": round(
                            scene_end_time
                            - scene_start_time,
                            2,
                        ),
                        "change_score": round(
                            difference,
                            4,
                        ),
                    }
                )

                print(
                    f"Scene {scene_number:3d} | "
                    f"{scene_start_time:7.2f}s → "
                    f"{scene_end_time:7.2f}s | "
                    f"Change: {difference:.3f}"
                )

                scene_number += 1

                scene_start_frame = (
                    frame_number
                )

                last_scene_change_time = (
                    current_time
                )

            previous_frame = frame

        # -----------------------------------------------------
        # FINAL SCENE
        # -----------------------------------------------------

        if total_frames > 0:

            scene_start_time = (
                scene_start_frame / fps
            )

            final_scene_end_time = (
                total_frames / fps
            )

            scenes.append(
                {
                    "scene_id": scene_number,
                    "start_frame": scene_start_frame,
                    "end_frame": total_frames,
                    "start_time": round(
                        scene_start_time,
                        2,
                    ),
                    "end_time": round(
                        final_scene_end_time,
                        2,
                    ),
                    "duration": round(
                        final_scene_end_time
                        - scene_start_time,
                        2,
                    ),
                    "change_score": 0.0,
                }
            )

        cap.release()

        # -----------------------------------------------------
        # FINAL SUMMARY
        # -----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("SCENE DETECTION COMPLETED")
        print("=" * 70)

        print(
            f"Total scenes detected : {len(scenes)}"
        )

        print(
            f"Video duration        : {duration:.2f} sec"
        )

        print("=" * 70)

        return {
            "video_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration": round(
                duration,
                2,
            ),
            "total_scenes": len(scenes),
            "scenes": scenes,
        }