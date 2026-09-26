import cv2
from ultralytics import YOLO

from app.vision.open_vocabulary import OpenVocabularyDetector
from app.intelligence.vision_enrichment import VisionEnricher


class SceneAnalyzer:
    """
    Scene visual analyzer.

    YOLO11:
        - People counting
        - General object detection

    OpenVocabularyDetector:
        - Additional object detection
        - microphone
        - piano
        - drum
        - guitar
        - etc.

    IMPORTANT:
        Detection counts are FRAME OBSERVATIONS.
        They are NOT unique-object counts.
    """

    # ------------------------------------------------------------
    # Objects that we don't want to report from YOLO11
    # because they are known unreliable for this project.
    # ------------------------------------------------------------

    IGNORED_OBJECTS = {
        "tie",
        "cell phone",
        "hat",
        "camera",
        "headset",
        "headphones",
        "earphones",
    }
    TEMPORAL_OBJECT_MINIMUMS = {
        "sports ball": 2,
        "water bottle": 3,
        "drum": 2,
        "frisbee": 2,
        "traffic light": 2,
        "window": 2,
        "desk": 2,
    }
    OBJECT_CONFIDENCE_MINIMUMS = {
        "water bottle": 0.60,
    }
    OBJECT_LABEL_ALIASES = {
        "bottle": "water bottle",
        "smartphone": "cell phone",
        "smart phone": "cell phone",
        "mobile phone": "cell phone",
        "cellphone": "cell phone",
    }

    def __init__(
        self,
        model_path="yolo11n.pt",
        confidence=0.50,
        sample_interval=10,
        open_vocabulary_confidence=0.25,
        vision_enricher=None,
    ):

        print("Loading YOLO11 for scene analysis...")

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.sample_interval = sample_interval
        self.open_vocabulary_confidence = (
            open_vocabulary_confidence
        )
        self.vision_enricher = (
            vision_enricher
            if vision_enricher is not None
            else VisionEnricher()
        )

        print("YOLO11 scene analyzer loaded.")

        # --------------------------------------------------------
        # Open Vocabulary detector
        # --------------------------------------------------------

        print(
            "Loading Open-Vocabulary detector..."
        )

        self.open_vocabulary = (
            OpenVocabularyDetector()
        )

        print(
            "Open-Vocabulary detector loaded."
        )

    @classmethod
    def normalize_object_label(cls, label):
        normalized = " ".join(str(label).lower().split())
        return cls.OBJECT_LABEL_ALIASES.get(normalized, normalized)

    @classmethod
    def meets_object_confidence_threshold(cls, label, confidence):
        minimum_confidence = cls.OBJECT_CONFIDENCE_MINIMUMS.get(
            cls.normalize_object_label(label),
            0.0,
        )
        return confidence >= minimum_confidence

    @classmethod
    def filter_temporally_unsupported_objects(
        cls,
        detection_observations,
        frame_observations,
    ):
        for object_name, minimum_frames in cls.TEMPORAL_OBJECT_MINIMUMS.items():
            supporting_frames = sum(
                observation["objects"].get(object_name, 0) > 0
                for observation in frame_observations
            )
            if supporting_frames < minimum_frames:
                detection_observations.pop(object_name, None)
                for observation in frame_observations:
                    observation["objects"].pop(object_name, None)

    # ============================================================
    # ANALYZE COMPLETE VIDEO
    # ============================================================

    def analyze_video(
        self,
        video_path,
        scenes,
    ):

        results = []

        for index, scene in enumerate(scenes):

            analyzed_scene = self.analyze_scene(
                video_path,
                scene,
                timeline_context={
                    "scene_index": index + 1,
                    "total_scenes": len(scenes),
                    "previous_scene": scenes[index - 1]
                    if index > 0
                    else None,
                    "next_scene": scenes[index + 1]
                    if index + 1 < len(scenes)
                    else None,
                },
            )

            if analyzed_scene is not None:

                results.append(
                    analyzed_scene
                )

        return results

    # ============================================================
    # ANALYZE ONE SCENE
    # ============================================================

    def analyze_scene(
        self,
        video_path,
        scene,
        timeline_context=None,
    ):

        scene_number = scene[
            "scene_number"
        ]

        start_time = float(
            scene["start_time"]
        )

        end_time = float(
            scene["end_time"]
        )

        print()
        print(
            f"Analyzing Scene #{scene_number}"
        )

        print(
            f"Time: "
            f"{start_time:.2f}s → "
            f"{end_time:.2f}s"
        )

        # --------------------------------------------------------
        # OPEN VIDEO
        # --------------------------------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
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

        # --------------------------------------------------------
        # FRAME RANGE
        # --------------------------------------------------------

        start_frame = max(
            0,
            int(start_time * fps)
        )

        end_frame = min(
            total_frames - 1,
            int(end_time * fps)
        )

        # --------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------

        sampled_frames = 0

        people_counts = []

        # Accumulated observations
        #
        # Example:
        #
        # person: 82
        # microphone: 12
        # piano: 8
        #
        # These are NOT unique objects.

        detection_observations = {}

        # Frame-level evidence

        frame_observations = []
        representative_frame = None

        # --------------------------------------------------------
        # MOVE TO SCENE START
        # --------------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        current_frame = start_frame

        # ========================================================
        # FRAME LOOP
        # ========================================================

        while current_frame <= end_frame:

            success, frame = cap.read()

            if not success:
                break

            relative_frame = (
                current_frame
                - start_frame
            )

            # ----------------------------------------------------
            # SAMPLE FRAME
            # ----------------------------------------------------

            if (
                relative_frame
                % self.sample_interval
                != 0
            ):

                current_frame += 1

                continue

            sampled_frames += 1
            representative_frame = frame.copy()

            timestamp = (
                current_frame / fps
            )

            # ====================================================
            # YOLO11
            # ====================================================

            prediction = (
                self.model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                )[0]
            )

            frame_people = 0

            frame_objects = {}

            # ----------------------------------------------------
            # PROCESS YOLO11 DETECTIONS
            # ----------------------------------------------------

            if prediction.boxes is not None:

                for box in prediction.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    object_name = self.normalize_object_label(
                        self.model.names[class_id]
                    )
                    object_name_lower = object_name

                    if not self.meets_object_confidence_threshold(
                        object_name_lower,
                        confidence,
                    ):
                        continue

                    # --------------------------------------------
                    # Ignore false detection
                    # --------------------------------------------

                    if (
                        object_name_lower
                        in self.IGNORED_OBJECTS
                    ):
                        continue

                    # --------------------------------------------
                    # PERSON
                    # --------------------------------------------

                    if (
                        object_name_lower
                        == "person"
                    ):

                        frame_people += 1

                    # --------------------------------------------
                    # Store YOLO object
                    # --------------------------------------------

                    frame_objects[
                        object_name
                    ] = (
                        frame_objects.get(
                            object_name,
                            0,
                        )
                        + 1
                    )

                    detection_observations[
                        object_name
                    ] = (
                        detection_observations.get(
                            object_name,
                            0,
                        )
                        + 1
                    )

            # ====================================================
            # OPEN-VOCABULARY DETECTION
            # ====================================================

            try:

                open_vocabulary_detections = (
                    self.open_vocabulary.detect(
                        frame,
                        confidence=(
                            self.open_vocabulary_confidence
                        ),
                    )
                )

            except Exception as error:

                print(
                    "Open-Vocabulary detection "
                    f"warning: {error}"
                )

                open_vocabulary_detections = []

            # ----------------------------------------------------
            # PROCESS OPEN-VOCABULARY OBJECTS
            # ----------------------------------------------------

            for detection in (
                open_vocabulary_detections
            ):

                label = self.normalize_object_label(
                    detection.get("label", "")
                )

                confidence = float(
                    detection.get(
                        "confidence",
                        0.0
                    )
                )
                if not self.meets_object_confidence_threshold(
                    label,
                    confidence,
                ):
                    continue

                # --------------------------------------------
                # Skip people here.
                #
                # YOLO11 remains the official
                # people counter.
                # --------------------------------------------

                if label in {
                    "person",
                    "man",
                    "woman",
                    "child",
                }:
                    continue

                # --------------------------------------------
                # Skip unwanted false object
                # --------------------------------------------

                if (
                    label
                    in self.IGNORED_OBJECTS
                ):
                    continue

                # --------------------------------------------
                # Store object observation
                # --------------------------------------------

                frame_objects[
                    label
                ] = (
                    frame_objects.get(
                        label,
                        0,
                    )
                    + 1
                )

                detection_observations[
                    label
                ] = (
                    detection_observations.get(
                        label,
                        0,
                    )
                    + 1
                )

            # ====================================================
            # PEOPLE STATISTICS
            # ====================================================

            people_counts.append(
                frame_people
            )

            # ====================================================
            # FRAME-LEVEL EVIDENCE
            # ====================================================

            frame_observations.append(
                {
                    "timestamp":
                        timestamp,

                    "people_count":
                        frame_people,

                    "objects":
                        frame_objects,
                }
            )

            current_frame += 1

        cap.release()

        # Require evidence in separate sampled frames, not duplicate model hits
        # from one frame, before including known false-positive-prone objects.
        self.filter_temporally_unsupported_objects(
            detection_observations,
            frame_observations,
        )

        # ========================================================
        # PEOPLE STATISTICS
        # ========================================================

        if people_counts:

            maximum_people = max(
                people_counts
            )

            average_people = (
                sum(people_counts)
                / len(people_counts)
            )

        else:

            maximum_people = 0

            average_people = 0.0

        enrichment = None
        if representative_frame is not None:
            enrichment = self.vision_enricher.enrich(
                representative_frame,
                {
                    "scene_number": scene_number,
                    "maximum_people": maximum_people,
                    "objects": detection_observations,
                    "timeline": timeline_context or {},
                },
            )

        if enrichment is not None:
            if enrichment.category:
                detection_category = enrichment.category
            else:
                detection_category = None
            enriched_actions = enrichment.actions
            enriched_emotion = enrichment.emotion
            narrative_type = enrichment.narrative_type
            narrative_confidence = enrichment.narrative_confidence
            narrative_evidence = enrichment.narrative_evidence
            dialogue_type = enrichment.dialogue_type
            dialogue_confidence = enrichment.dialogue_confidence
            dialogue_evidence = enrichment.dialogue_evidence
            enriched_scene_labels = enrichment.scene_labels
            appearance_observations = enrichment.appearance
            enriched_description = enrichment.description
            enriched_objects = enrichment.objects
        else:
            detection_category = None
            enriched_actions = []
            enriched_emotion = None
            narrative_type = None
            narrative_confidence = None
            narrative_evidence = []
            dialogue_type = None
            dialogue_confidence = None
            dialogue_evidence = []
            enriched_scene_labels = []
            appearance_observations = {}
            enriched_description = ""
            enriched_objects = []

        # ========================================================
        # PRINT RESULT
        # ========================================================

        print()

        print(
            "Maximum people in one frame: "
            f"{maximum_people}"
        )

        print(
            "Average people per frame: "
            f"{average_people:.2f}"
        )

        print()

        print(
            "Frame detection observations:"
        )

        if detection_observations:

            for (
                object_name,
                count,
            ) in sorted(
                detection_observations.items()
            ):

                print(
                    f"  - {object_name}: "
                    f"{count}"
                )

        else:

            print(
                "  - None"
            )

        # ========================================================
        # RETURN RESULT
        # ========================================================

        return {

            "scene_number":
                scene_number,

            "start_time":
                start_time,

            "end_time":
                end_time,

            "duration":
                round(
                    end_time - start_time,
                    2,
                ),

            "sampled_frames":
                sampled_frames,

            "maximum_people":
                maximum_people,

            "average_people":
                round(
                    average_people,
                    2,
                ),
            "category": detection_category,
            "actions": enriched_actions,
            "emotion": enriched_emotion,
            "narrative_type": narrative_type,
            "narrative_confidence": narrative_confidence,
            "narrative_evidence": narrative_evidence,
            "dialogue_type": dialogue_type,
            "dialogue_confidence": dialogue_confidence,
            "dialogue_evidence": dialogue_evidence,
            "scene_labels": enriched_scene_labels,
            "appearance_observations": appearance_observations,
            "vision_description": enriched_description,
            "vision_objects": enriched_objects,

            # ----------------------------------------------------
            # Accumulated frame observations
            # ----------------------------------------------------

            "objects":
                detection_observations,

            "frame_detection_observations":
                detection_observations,

            # ----------------------------------------------------
            # Individual sampled-frame evidence
            # ----------------------------------------------------

            "frame_observations":
                frame_observations,
        }