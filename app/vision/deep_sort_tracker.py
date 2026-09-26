from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortPersonTracker:

    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False
        )

    def update(self, detections, frame):
        """
        detections format:

        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.92
            }
        ]
        """

        tracker_detections = []

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            tracker_detections.append(
                (
                    [x1, y1, width, height],
                    detection["confidence"],
                    "person"
                )
            )

        tracks = self.tracker.update_tracks(
            tracker_detections,
            frame=frame
        )

        results = []

        for track in tracks:

            if not track.is_confirmed():
                continue

            if track.time_since_update > 0:
                continue

            track_id = track.track_id

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                ltrb
            )

            results.append(
    {
        "track_id": track_id,
        "label": "person",
        "bbox": [
            x1,
            y1,
            x2,
            y2
        ],
        "confidence": float(
            track.det_conf
        ) if track.det_conf is not None else 0.0,
    }
)
        return results