"""Time-range queries over analyzed video scenes."""


def find_scenes_in_range(scenes, start_time, end_time):
    """Return scenes that overlap the inclusive requested time range."""
    try:
        start = float(start_time)
        end = float(end_time)
    except (TypeError, ValueError) as error:
        raise ValueError("Time range values must be numeric.") from error

    if start < 0 or end < 0:
        raise ValueError("Time range values cannot be negative.")
    if end <= start:
        raise ValueError("The end time must be greater than the start time.")

    matches = []
    for scene in scenes:
        if not isinstance(scene, dict):
            raise TypeError("Each scene must be represented by a dictionary.")

        scene_start = float(scene.get("start_time", 0.0))
        scene_end = float(scene.get("end_time", scene_start))

        if scene_start < end and scene_end > start:
            matches.append(scene)

    return matches


def describe_time_range(scenes, start_time, end_time):
    """Build a concise answer for a requested time range."""
    matches = find_scenes_in_range(scenes, start_time, end_time)
    if not matches:
        return {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "scenes": [],
            "description": "No analyzed scene overlaps this time range.",
        }

    descriptions = []
    for scene in matches:
        description = (
            scene.get("scene_description")
            or scene.get("description")
            or scene.get("event_description")
        )
        if description:
            descriptions.append(str(description))

    return {
        "start_time": float(start_time),
        "end_time": float(end_time),
        "scenes": matches,
        "description": " ".join(descriptions)
        if descriptions
        else "The selected range overlaps analyzed scenes, but no description is available.",
    }
