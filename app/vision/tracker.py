from collections import defaultdict


class ObjectTracker:

    def __init__(self):

        # Stores track IDs grouped by object label
        self.tracks = defaultdict(set)

        # Stores the latest information for each track
        self.track_objects = {}

    def update(self, detections):

        """
        Update the tracker using detections produced
        by the YOLO detector.

        Each detection should contain:

            label
            confidence
            bbox
            track_id
        """

        current_objects = []

        for detection in detections:

            label = detection.get("label")

            confidence = detection.get(
                "confidence",
                0.0
            )

            bbox = detection.get(
                "bbox"
            )

            track_id = detection.get(
                "track_id"
            )

            # Ignore detections without tracking IDs
            if track_id is None:
                continue

            # Store track ID under object category
            self.tracks[label].add(
                track_id
            )

            # Store latest object information
            self.track_objects[track_id] = {

                "track_id": track_id,

                "label": label,

                "confidence": confidence,

                "bbox": bbox
            }

            current_objects.append(
                self.track_objects[track_id]
            )

        return current_objects


    def get_unique_counts(self):

        """
        Return the number of unique tracked
        objects for every object category.
        """

        counts = {}

        for label, track_ids in self.tracks.items():

            counts[label] = len(track_ids)

        return counts


    def get_all_tracks(self):

        """
        Return all currently known tracks.
        """

        return list(
            self.track_objects.values()
        )


    def reset(self):

        """
        Clear all tracking information.
        """

        self.tracks.clear()

        self.track_objects.clear()