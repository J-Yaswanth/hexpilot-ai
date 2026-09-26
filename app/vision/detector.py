from ultralytics import YOLO


class VideoObjectDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame, confidence=0.35):
        results = self.model(
            frame,
            conf=confidence,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])

                class_name = self.model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append({
                    "label": class_name,
                    "confidence": confidence_score,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections

    def track(self, frame, confidence=0.35):
        results = self.model.track(
            frame,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])

                class_name = self.model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                detections.append({
                    "label": class_name,
                    "confidence": confidence_score,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id
                })

        return detections