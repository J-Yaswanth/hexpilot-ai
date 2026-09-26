class SceneIntelligence:

    def __init__(self, event_analyzer):
        """
        Combines:
        SceneAnalyzer output
        +
        EventAnalyzer output

        into a final human-readable scene description.
        """

        self.event_analyzer = event_analyzer

    # ---------------------------------------------------------
    # ANALYZE SCENE
    # ---------------------------------------------------------

    def analyze_scene(self, scene):

        people = scene.get(
            "maximum_people",
            0
        )

        objects = scene.get(
            "objects",
            {}
        )

        # -----------------------------------------------------
        # EVENT ANALYSIS
        # -----------------------------------------------------

        event_input = {
            "people": people,
            "objects": objects,
        }

        event_result = self.event_analyzer.analyze(
            event_input
        )

        # -----------------------------------------------------
        # EXTRACT EVENT INFORMATION
        # -----------------------------------------------------

        event_type = event_result.get(
            "event_type",
            "unknown"
        )

        events = event_result.get(
            "events",
            []
        )

        description = event_result.get(
            "description",
            ""
        )
        category = (
            scene.get("category")
            or event_result.get("category")
            or "other"
        )
        location = (
            scene.get("location")
            or event_result.get("location")
            or "unknown"
        )
        evidence_confidence = scene.get(
            "evidence_confidence",
            event_result.get("evidence_confidence", 0.0),
        )
        narrative_type = scene.get(
            "narrative_type",
            "unknown",
        ) or "unknown"
        narrative_confidence = scene.get(
            "narrative_confidence",
            None,
        )
        narrative_evidence = scene.get(
            "narrative_evidence",
            [],
        )
        dialogue_type = scene.get(
            "dialogue_type",
            "unknown",
        ) or "unknown"
        dialogue_confidence = scene.get(
            "dialogue_confidence",
            None,
        )
        dialogue_evidence = scene.get(
            "dialogue_evidence",
            [],
        )
        scene_labels = scene.get(
            "scene_labels",
            [],
        )
        actions = scene.get(
            "actions",
            event_result.get("actions", []),
        )
        emotion = scene.get("emotion") or "not confidently determined"
        vision_description = scene.get(
            "vision_description",
            "",
        )

        # -----------------------------------------------------
        # OBJECT LIST
        # -----------------------------------------------------

        detected_objects = []

        for object_name in objects:

            if object_name.lower() == "person":
                continue

            detected_objects.append(
                object_name
            )

        detected_objects.sort()

        # -----------------------------------------------------
        # BUILD FINAL DESCRIPTION
        # -----------------------------------------------------

        final_description_parts = []

        # Event
        if event_type != "unknown":

            final_description_parts.append(
                f"The scene appears to show "
                f"{event_type}."
            )

        # People
        if people > 0:

            final_description_parts.append(
                f"There are approximately "
                f"{people} people visible."
            )

        # Objects
        if detected_objects:

            object_text = ", ".join(
                detected_objects
            )

            final_description_parts.append(
                f"Detected objects include "
                f"{object_text}."
            )

        # Event details
        for event in events:

            if event not in final_description_parts:

                final_description_parts.append(
                    event
                )

        # Fallback
        if not final_description_parts:

            final_description_parts.append(
                "No significant activity "
                "could be determined."
            )

        final_description = " ".join(
            final_description_parts
        )

        if vision_description:
            final_description = vision_description
        elif description:
            final_description = description

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        return {

            "scene_number":
                scene.get(
                    "scene_number"
                ),

            "start_time":
                scene.get(
                    "start_time"
                ),

            "end_time":
                scene.get(
                    "end_time"
                ),

            "duration":
                scene.get(
                    "duration"
                ),

            "people":
                people,

            "objects":
                objects,

            "detected_objects":
                detected_objects,

            "event_type":
                event_type,

            "category":
                category,

            "location":
                location,

            "evidence_confidence":
                evidence_confidence,

            "narrative_type":
                narrative_type,

            "narrative_confidence":
                narrative_confidence,

            "narrative_evidence":
                narrative_evidence,

            "dialogue_type":
                dialogue_type,

            "dialogue_confidence":
                dialogue_confidence,

            "dialogue_evidence":
                dialogue_evidence,

            "scene_labels":
                scene_labels,

            "actions":
                actions,

            "emotion":
                emotion,

            "events":
                events,

            "event_description":
                description,

            "description":
                final_description,

            "scene_description":
                final_description,
        }

    # ---------------------------------------------------------
    # ANALYZE ALL SCENES
    # ---------------------------------------------------------

    def analyze_scenes(
        self,
        scenes
    ):

        results = []

        for scene in scenes:

            result = self.analyze_scene(
                scene
            )

            results.append(
                result
            )

        return results