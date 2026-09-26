class TemporalPersonAnalyzer:

    def __init__(
        self,
        min_observations=3,
        confidence_threshold=0.50,
    ):
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold

        self.frame_history = []
        self.total_frames = 0

        self.person_observations = 0
        self.detection_counts = []
        self.confidence_history = []

    # ---------------------------------------------------------
    # ADD FRAME
    # ---------------------------------------------------------

    def add_frame(self, detections):

        self.total_frames += 1

        valid_people = []

        for detection in detections:

            label = detection.get(
                "label",
                ""
            ).lower()

            confidence = float(
                detection.get(
                    "confidence",
                    0.0
                )
            )

            if label != "person":
                continue

            if confidence < self.confidence_threshold:
                continue

            valid_people.append(detection)

            self.person_observations += 1

            self.confidence_history.append(
                confidence
            )

        self.detection_counts.append(
            len(valid_people)
        )

        self.frame_history.append(
            valid_people
        )

    # ---------------------------------------------------------
    # STATISTICS
    # ---------------------------------------------------------

    def get_detection_statistics(self):

        if not self.detection_counts:

            return {
                "frames_analyzed": 0,
                "minimum": 0,
                "maximum": 0,
                "average": 0.0,
                "person_observations": 0,
            }

        return {
            "frames_analyzed":
                self.total_frames,

            "minimum":
                min(self.detection_counts),

            "maximum":
                max(self.detection_counts),

            "average":
                round(
                    sum(self.detection_counts)
                    / len(self.detection_counts),
                    2
                ),

            "person_observations":
                self.person_observations,
        }

    # ---------------------------------------------------------
    # ESTIMATE PEOPLE
    # ---------------------------------------------------------

    def estimate_person_presence(self):

        if not self.detection_counts:

            return {
                "estimated_people": 0,
                "confidence": 0.0,
            }

        maximum_people = max(
            self.detection_counts
        )

        # Count frames where the maximum
        # number of people was detected.
        supporting_frames = sum(
            1
            for count in self.detection_counts
            if count == maximum_people
        )

        total_frames = len(
            self.detection_counts
        )

        temporal_support = (
            supporting_frames
            / total_frames
        )

        if self.confidence_history:

            average_confidence = (
                sum(
                    self.confidence_history
                )
                /
                len(
                    self.confidence_history
                )
            )

        else:

            average_confidence = 0.0

        evidence_confidence = (
            temporal_support
            * average_confidence
        )

        return {
            "estimated_people":
                maximum_people,

            "confidence":
                round(
                    evidence_confidence,
                    3
                ),

            "temporal_support":
                round(
                    temporal_support,
                    3
                ),

            "average_detection_confidence":
                round(
                    average_confidence,
                    3
                ),

            "supporting_frames":
                supporting_frames,

            "total_frames":
                total_frames,
        }

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    def get_summary(self):

        return {
            "detection_statistics":
                self.get_detection_statistics(),

            "person_presence":
                self.estimate_person_presence(),
        }