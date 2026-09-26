class EventAnalyzer:

    def __init__(self):
        pass

    # =========================================================
    # ANALYZE SCENE
    # =========================================================

    def analyze_scene(self, scene, objects=None):
        """
        Analyze a scene.

        Supports:

            analyze_scene(scene)

        or:

            analyze_scene(people_count, objects)
        """

        # -----------------------------------------------------
        # FORMAT 1:
        # analyze_scene(scene_dict)
        # -----------------------------------------------------

        if isinstance(scene, dict):

            people_count = scene.get(
                "maximum_people",
                scene.get("people", 0)
            )

            if objects is None:
                objects = scene.get(
                    "objects",
                    {}
                )

        # -----------------------------------------------------
        # FORMAT 2:
        # analyze_scene(people_count, objects)
        # -----------------------------------------------------

        else:

            people_count = scene

        # -----------------------------------------------------
        # PEOPLE
        # -----------------------------------------------------

        try:
            people_count = int(
                people_count
            )
        except (TypeError, ValueError):
            people_count = 0

        # -----------------------------------------------------
        # OBJECTS
        # -----------------------------------------------------

        if not isinstance(objects, dict):
            objects = {}

        normalized_objects = {}

        for name, count in objects.items():

            name = str(
                name
            ).lower().strip()

            try:
                count = int(count)
            except (TypeError, ValueError):
                continue

            # Person is handled separately.
            if name == "person":
                continue

            normalized_objects[name] = count

        object_names = set(
            normalized_objects.keys()
        )

        # =====================================================
        # EVENT DETECTION
        # =====================================================

        event_type = "unknown"
        category = "other"
        location = "unknown"
        actions = []

        events = []

        # -----------------------------------------------------
        # MUSIC / PERFORMANCE
        # -----------------------------------------------------

        music_objects = {
            "microphone",
            "piano",
            "drum",
            "guitar",
            "violin",
            "keyboard",
            "trumpet",
            "cello",
            "flute",
        }

        music_found = (
            object_names
            .intersection(music_objects)
        )

        strong_music_evidence = (
            "microphone" in object_names
            or len(music_found) >= 2
        )

        gym_objects = {
            "dumbbell",
            "barbell",
            "weight plate",
            "kettlebell",
            "bench press",
            "gym bench",
            "squat rack",
            "power rack",
            "treadmill",
            "exercise bike",
            "rowing machine",
            "cable machine",
            "pull-up bar",
            "gym equipment",
        }

        gym_found = object_names.intersection(gym_objects)

        location_rules = (
            (
                "gym",
                gym_objects,
            ),
            (
                "hospital",
                {"hospital bed", "medical equipment"},
            ),
            (
                "school / classroom",
                {"school desk", "classroom"},
            ),
            (
                "police setting",
                {"police uniform"},
            ),
            (
                "restaurant / dining area",
                {"restaurant table"},
            ),
            (
                "kitchen",
                {"kitchen counter"},
            ),
            (
                "park / outdoor",
                {"park bench", "forest", "beach"},
            ),
            (
                "office",
                {"laptop", "computer", "desk", "keyboard"},
            ),
            (
                "kitchen",
                {"pot", "pan", "frying pan", "cooking utensil"},
            ),
            (
                "restaurant / dining area",
                {"plate", "bowl", "fork", "spoon", "food"},
            ),
            (
                "road / street",
                {"car", "bus", "truck", "motorcycle", "traffic light"},
            ),
        )

        for candidate_location, evidence in location_rules:
            if object_names.intersection(evidence):
                location = candidate_location
                break

        cooking_objects = {
            "pot",
            "pan",
            "frying pan",
            "cooking utensil",
        }
        eating_objects = {
            "plate",
            "bowl",
            "fork",
            "spoon",
            "knife",
            "food",
        }
        drinking_objects = {
            "cup",
            "glass",
            "water bottle",
        }
        dance_objects = {
            "microphone",
            "stage",
            "dance floor",
        }
        driving_objects = {
            "car",
            "motorcycle",
            "steering wheel",
        }

        if people_count > 0 and gym_found:

            event_type = "fitness / physical activity"
            category = "action"
            actions.append("strength training or gym workout")
            events.append("A person is working out in a gym")

            for equipment in sorted(gym_found):
                events.append(
                    f"{equipment.capitalize()} equipment is present"
                )

        elif music_found and strong_music_evidence:

            event_type = (
                "music / performance"
            )
            category = "music"

            events.append(
                "Possible musical performance"
            )

            if "microphone" in object_names and people_count > 0:
                actions.append("singing or speaking into a microphone")

            if "microphone" in object_names:

                events.append(
                    "A microphone is present"
                )

            if "piano" in object_names:

                events.append(
                    "A piano is present"
                )

            if "drum" in object_names:

                events.append(
                    "A drum is present"
                )

            if "guitar" in object_names:

                events.append(
                    "A guitar is present"
                )

        elif people_count > 0 and object_names.intersection(
            cooking_objects
        ):

            event_type = "cooking / food preparation"
            category = "daily life"
            actions.append("cooking or preparing food")
            events.append("Food preparation equipment is present")

        elif people_count > 0 and object_names.intersection(
            eating_objects
        ):

            event_type = "eating / meal"
            category = "daily life"
            actions.append("eating or handling food")
            events.append("Food or dining items are present")

        elif people_count > 0 and object_names.intersection(
            drinking_objects
        ):

            event_type = "drinking"
            category = "daily life"
            actions.append("drinking or holding a beverage")
            events.append("A drinking container is present")

        elif people_count > 0 and object_names.intersection(
            dance_objects
        ) and "microphone" not in object_names:

            event_type = "dance / performance"
            category = "music"
            actions.append("dancing or performing")
            events.append("A performance or dance setting is present")

        elif object_names.intersection(driving_objects):

            event_type = "driving / transportation"
            category = "transportation"
            actions.append("driving or riding in a vehicle")
            events.append("A vehicle is present")

        elif object_names.intersection({
            "smartphone",
            "laptop",
            "computer",
        }):

            event_type = "technology use"
            category = "daily life"
            actions.append("using a phone or computer")
            events.append("A personal device is present")

        # -----------------------------------------------------
        # SPORTS
        # -----------------------------------------------------

        elif (
            people_count > 0
            and object_names.intersection({
                "water bottle",
                "sports ball",
                "baseball bat",
                "tennis racket",
                "skateboard",
                "surfboard",
            })
        ):

            event_type = "fitness / physical activity"
            category = "action"
            actions.append("running or physical activity")

            if "water bottle" in object_names:
                events.append("A person may be drinking water")

            if object_names.intersection({
                "headset",
                "headphones",
                "earphones",
            }):
                events.append("A headset or earphones are present")

        elif (
            object_names.intersection({
                "sports ball",
                "baseball bat",
                "tennis racket",
                "skateboard",
                "surfboard",
            })
            and not object_names.intersection({
                "car",
                "bus",
                "truck",
                "motorcycle",
                "bicycle",
                "train",
                "boat",
                "airplane",
                "van",
            })
        ):

            event_type = (
                "sports / activity"
            )
            category = "action"
            actions.append("sports or physical activity")

            events.append(
                "Possible sports or physical activity"
            )

        # -----------------------------------------------------
        # TRANSPORTATION
        # -----------------------------------------------------

        elif object_names.intersection({
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "train",
            "boat",
            "airplane",
        }):

            event_type = (
                "transportation"
            )
            category = "transportation"

            events.append(
                "A transportation-related scene is detected"
            )

        # -----------------------------------------------------
        # WORK / ACTIVITY
        # -----------------------------------------------------

        elif object_names.intersection({
            "laptop",
            "computer",
            "keyboard",
            "cell phone",
        }):

            event_type = (
                "work / activity"
            )
            category = "work"

            events.append(
                "People appear to be engaged in a work or activity scene"
            )

        # -----------------------------------------------------
        # PEOPLE ONLY
        # -----------------------------------------------------

        elif people_count >= 2:

            event_type = "conversation / interaction"
            category = "interaction"
            actions.append("people interacting or conversing")

            events.append(
                "Multiple people are present and may be interacting"
            )

        elif people_count > 0:

            event_type = "people / interaction"
            category = "interaction"

            events.append(
                f"{people_count} people are present in the scene"
            )

        # -----------------------------------------------------
        # UNKNOWN
        # -----------------------------------------------------

        else:

            event_type = "unknown"

            events.append(
                "No significant activity detected"
            )

        # =====================================================
        # DESCRIPTION
        # =====================================================

        description_parts = []

        if event_type != "unknown":

            description_parts.append(
                f"The scene appears to show "
                f"{event_type}."
            )

        if people_count > 0:

            description_parts.append(
                f"A maximum of {people_count} "
                f"people are visible in a sampled frame."
            )

        if normalized_objects:

            object_names = sorted(
                normalized_objects.keys()
            )
            if len(object_names) > 1:
                object_text = (
                    ", ".join(object_names[:-1])
                    + ", and "
                    + object_names[-1]
                )
            else:
                object_text = object_names[0]

            description_parts.append(
                "Objects detected in the sampled frames "
                f"include {object_text}."
            )

        if events:
            primary_event = events[0].rstrip(".")
            description_parts.append(f"{primary_event}.")
            if len(events) > 1:
                details = ", ".join(
                    event.rstrip(".").lower()
                    for event in events[1:]
                )
                description_parts.append(
                    f"Additional details: {details}."
                )

        description = " ".join(
            description_parts
        )

        if not description:

            description = (
                "The scene could not be confidently interpreted."
            )

        evidence_confidence = 0.0
        if event_type != "unknown":
            evidence_confidence = 0.6
            if normalized_objects:
                evidence_confidence += 0.1
            if people_count > 0:
                evidence_confidence += 0.1
            if len(events) > 1:
                evidence_confidence += 0.1
            evidence_confidence = min(
                round(evidence_confidence, 2),
                0.9,
            )

        # =====================================================
        # RESULT
        # =====================================================

        return {

             "event_type": event_type,
    "category": category,
    "location": location,
    "evidence_confidence": evidence_confidence,
    "actions": actions,
    "appearance_observations": [],

    # Keep both names for compatibility
    "people": people_count,
    "people_count": people_count,

    "events": events,

    "description": description,

    "detected_objects": sorted(
        normalized_objects.keys()
    ),

    "object_counts": normalized_objects,
        }

    # =========================================================
    # COMPATIBILITY
    # =========================================================

    def analyze(
        self,
        scene,
        objects=None
    ):

        return self.analyze_scene(
            scene,
            objects
        )