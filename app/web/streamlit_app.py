from pathlib import Path
import sys
import subprocess
import tempfile
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import streamlit as st
from imageio_ffmpeg import get_ffmpeg_exe

from app.intelligence.scene_detector import SceneDetector
from app.intelligence.scene_analyzer import SceneAnalyzer
from app.intelligence.event_analyzer import EventAnalyzer
from app.intelligence.scene_intelligence import SceneIntelligence
from app.intelligence.time_query import describe_time_range


def _format_timestamp(total_seconds):
    total_seconds = max(0.0, float(total_seconds))
    minutes = int(total_seconds // 60)
    seconds = total_seconds - (minutes * 60)
    return f"{minutes:02d}:{seconds:05.2f}"


def _extract_video_range(video_path, start_time, end_time):
    output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
    )
    output_path = output.name
    output.close()

    duration = float(end_time) - float(start_time)
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-ss",
        str(float(start_time)),
        "-i",
        str(video_path),
        "-t",
        str(duration),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(
            "The video encoder could not be started."
        ) from error

    if process.returncode != 0:
        raise RuntimeError(
            "Could not create the selected video range: "
            + process.stderr[-1000:]
        )

    return output_path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Video Intelligence Studio",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# CUSTOM FRONT-END STYLE
# ============================================================

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #0e1117;
        color: #f3f4f6;
    }

    /* Keep text readable against the custom dark background */
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp span,
    .stApp div[data-testid="stMarkdownContainer"] {
        color: #f3f4f6;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #f8fafc;
    }

    .subtitle {
        color: #b8bcc8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Section cards */
    .section-card {
        background: #171a21;
        border: 1px solid #2d313b;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
    }

    .scene-card {
        background: #171a21;
        border: 1px solid #303541;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .scene-header {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .metric-label {
        color: #aeb4c0;
        font-size: 0.9rem;
    }

    .metric-value {
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #171a21;
        border-radius: 10px;
        padding: 0.5rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #20242d;
        border-color: #454b57;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #f3f4f6;
    }

    /* Primary Analyze button */
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #ff3333;
        border-color: #ff3333;
        color: white;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #1f9bf0;
    }

    /* Info/success/error boxes */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Reduce excessive top spacing */
    h1, h2, h3 {
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI Video Intelligence Studio</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Scene Detection & Video Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Upload a video clip and the system will analyze:

    - 🎬 Scene boundaries
    - ⏱️ Scene timestamps
    - 👤 People
    - 🔎 Objects
    - 🎭 Scene/event type
    - 📝 Scene description
    """
)


# ============================================================
# ANALYSIS DEFAULTS
# ============================================================

confidence = 0.50
sample_interval = 10


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a video clip",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
    ],
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # SAVE TEMPORARY VIDEO
    # --------------------------------------------------------

    suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_video_path = temp_file.name

    # --------------------------------------------------------
    # SHOW ORIGINAL VIDEO
    # --------------------------------------------------------

    st.subheader("🎥 Uploaded Video")

    st.video(
        uploaded_file
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    analyze_button = st.button(
        "🚀 Analyze Video",
        type="primary",
        width="stretch",
    )

    if analyze_button:

        progress = st.progress(0)
        status = st.empty()

        try:

            # ==================================================
            # STEP 1 — SCENE DETECTION
            # ==================================================

            status.write(
                "🎬 Detecting scenes..."
            )

            progress.progress(20)

            scene_detector = SceneDetector()

            scenes_result = scene_detector.detect_scenes(
                temp_video_path
            )

            # --------------------------------------------------
            # NORMALIZE SceneDetector OUTPUT
            # --------------------------------------------------

            if isinstance(
                scenes_result,
                dict,
            ):

                if "scenes" in scenes_result:

                    scenes = scenes_result["scenes"]

                elif "scene_results" in scenes_result:

                    scenes = scenes_result["scene_results"]

                else:

                    raise ValueError(
                        "SceneDetector returned a dictionary, "
                        "but no 'scenes' or 'scene_results' key "
                        "was found."
                    )

            elif isinstance(
                scenes_result,
                list,
            ):

                scenes = scenes_result

            else:

                raise TypeError(
                    "Unexpected SceneDetector result type: "
                    f"{type(scenes_result)}"
                )

            if not scenes:

                st.warning(
                    "No scenes were detected."
                )

                progress.progress(100)

                status.write(
                    "Scene detection completed."
                )

                st.stop()

            # --------------------------------------------------
            # NORMALIZE SCENE FORMAT
            # --------------------------------------------------

            normalized_scenes = []

            for index, scene in enumerate(
                scenes,
                start=1,
            ):

                if isinstance(
                    scene,
                    dict,
                ):

                    start_time = scene.get(
                        "start_time",
                        scene.get(
                            "start",
                            0.0,
                        ),
                    )

                    end_time = scene.get(
                        "end_time",
                        scene.get(
                            "end",
                            0.0,
                        ),
                    )

                    scene_number = scene.get(
                        "scene_number",
                        index,
                    )

                elif isinstance(
                    scene,
                    (list, tuple),
                ):

                    if len(scene) < 2:

                        raise ValueError(
                            f"Invalid scene format: {scene}"
                        )

                    scene_number = index
                    start_time = scene[0]
                    end_time = scene[1]

                else:

                    raise TypeError(
                        "Invalid scene item type: "
                        f"{type(scene)}"
                    )

                normalized_scenes.append(
                    {
                        "scene_number": int(
                            scene_number
                        ),
                        "start_time": float(
                            start_time
                        ),
                        "end_time": float(
                            end_time
                        ),
                    }
                )

            scenes = normalized_scenes

            # --------------------------------------------------
            # SHOW SCENE TIMELINE
            # --------------------------------------------------

            st.subheader("🎬 Scene Timeline")

            st.write(
                f"Detected {len(scenes)} scene(s)."
            )

            for scene in scenes:

                st.write(
                    f"**Scene #{scene['scene_number']}** "
                    f"— "
                    f"{scene['start_time']:.2f}s "
                    f"→ "
                    f"{scene['end_time']:.2f}s"
                )

            progress.progress(35)

            # ==================================================
            # STEP 2 — VISUAL ANALYSIS
            # ==================================================

            status.write(
                "👁️ Detecting people and objects..."
            )

            progress.progress(45)

            scene_analyzer = SceneAnalyzer(
                confidence=confidence,
                sample_interval=sample_interval,
            )

            analyzed_scenes = scene_analyzer.analyze_video(
                temp_video_path,
                scenes,
            )

            progress.progress(65)

            # ==================================================
            # STEP 3 — SCENE INTELLIGENCE
            # ==================================================

            status.write(
                "🧠 Understanding scene activities..."
            )

            event_analyzer = EventAnalyzer()

            # IMPORTANT:
            # The current SceneIntelligence class expects
            # one dependency, EventAnalyzer.
            intelligence = SceneIntelligence(
                event_analyzer
            )

            intelligent_scenes = []

            for scene in analyzed_scenes:

                intelligence_result = (
                    intelligence.analyze_scene(
                        scene
                    )
                )

                combined_scene = dict(
                    scene
                )

                if isinstance(
                    intelligence_result,
                    dict,
                ):

                    combined_scene.update(
                        intelligence_result
                    )

                intelligent_scenes.append(
                    combined_scene
                )

            progress.progress(85)

            # ==================================================
            # SAVE RESULTS IN SESSION
            # ==================================================

            st.session_state[
                "video_analysis"
            ] = intelligent_scenes

            # ==================================================
            # DISPLAY RESULTS
            # ==================================================

            st.subheader(
                "📊 Video Analysis Results"
            )

            for scene in intelligent_scenes:

                scene_number = scene.get(
                    "scene_number",
                    "?",
                )

                start_time = float(
                    scene.get(
                        "start_time",
                        0.0,
                    )
                )

                end_time = float(
                    scene.get(
                        "end_time",
                        0.0,
                    )
                )

                st.markdown(
                    f'<div class="scene-header">'
                    f"🎬 Scene #{scene_number}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.write(
                    f"⏱️ **Time:** "
                    f"{start_time:.2f}s → "
                    f"{end_time:.2f}s"
                )

                # ------------------------------------------------
                # PEOPLE
                # ------------------------------------------------

                maximum_people = scene.get(
                    "maximum_people",
                    scene.get(
                        "people",
                        0,
                    ),
                )

                average_people = scene.get(
                    "average_people",
                    0,
                )

                st.markdown(
                    f"👥 **Maximum people:** "
                    f"{maximum_people}"
                )

                st.markdown(
                    f"👥 **Average people:** "
                    f"{average_people}"
                )

                # ------------------------------------------------
                # EVENT TYPE
                # ------------------------------------------------

                event_type = scene.get(
                    "event_type",
                    "Unknown",
                )

                st.markdown(
                    f"🎭 **Event type:** "
                    f"{event_type}"
                )

                category = scene.get(
                    "category",
                    "other"
                ) or "other"
                st.markdown(
                    f"🏷️ **Category:** "
                    f"{category}"
                )

                location = scene.get(
                    "location",
                    "unknown",
                ) or "unknown"
                if location != "unknown":
                    st.markdown(
                        f"📍 **Location:** "
                        f"{location}"
                    )

                confidence_score = float(
                    scene.get(
                        "evidence_confidence",
                        0.0,
                    )
                )
                st.markdown(
                    f"📊 **Evidence confidence:** "
                    f"{confidence_score:.0%}"
                )

                narrative_type = scene.get(
                    "narrative_type",
                    "unknown",
                ) or "unknown"
                if narrative_type != "unknown":
                    st.markdown(
                        f"📖 **Narrative scene:** "
                        f"{narrative_type}"
                    )

                dialogue_type = scene.get(
                    "dialogue_type",
                    "unknown",
                ) or "unknown"
                if dialogue_type != "unknown":
                    st.markdown(
                        f"💬 **Dialogue scene:** "
                        f"{dialogue_type}"
                    )

                scene_labels = scene.get("scene_labels", [])
                if scene_labels:
                    st.markdown("### 🎞️ Scene labels")
                    for scene_label in scene_labels:
                        groups = scene_label.get(
                            "groups",
                            [scene_label["group"]],
                        )
                        st.markdown(
                            f"- **{scene_label['label']}** "
                            f"({', '.join(groups)}, "
                            f"{scene_label['confidence']:.0%})"
                        )
                        if scene_label["evidence"]:
                            st.caption(
                                "; ".join(scene_label["evidence"])
                            )

                actions = scene.get(
                    "actions",
                    [],
                )
                if actions:
                    st.markdown("### 🎬 Actions")
                    for action in actions:
                        st.write(f"• {action}")

                # ------------------------------------------------
                # DESCRIPTION
                # ------------------------------------------------

                description = scene.get(
                    "description",
                    "",
                )

                if description:

                    st.markdown(
                        "### 📝 Scene Description"
                    )

                    st.info(
                        description
                    )

                # ------------------------------------------------
                # EVENTS
                # ------------------------------------------------

                events = scene.get(
                    "events",
                    [],
                )

                if events:

                    st.markdown(
                        "### 🎯 Events"
                    )

                    for event in events:

                        st.write(
                            f"• {event}"
                        )

                # ------------------------------------------------
                # OBJECTS
                # ------------------------------------------------

                objects = scene.get(
                    "objects",
                    {},
                )

                if objects:

                    st.markdown(
                        "### 🔎 Detected Objects"
                    )

                    for (
                        object_name,
                        count,
                    ) in objects.items():

                        st.write(
                            f"• **{object_name}** "
                            f"— {count} detections"
                        )

                appearance = scene.get(
                    "appearance_observations",
                    {},
                )
                if appearance:
                    st.markdown("### 👗 Appearance and visual details")
                    st.json(appearance)

                vision_objects = scene.get(
                    "vision_objects",
                    [],
                )
                if vision_objects:
                    st.markdown("### 🧠 Vision-model observations")
                    st.json(vision_objects)

                st.divider()

            # ==================================================
            # TIMELINE TABLE
            # ==================================================

            st.markdown("---")

            st.subheader(
                "⏱️ Scene Timeline"
            )

            timeline_rows = []

            for scene in intelligent_scenes:

                timeline_rows.append(
                    {
                        "Scene":
                            scene.get(
                                "scene_number",
                                "",
                            ),

                        "Start":
                            f"{scene.get('start_time', 0):.2f}s",

                        "End":
                            f"{scene.get('end_time', 0):.2f}s",

                        "Event":
                            scene.get(
                                "event_type",
                                "Unknown",
                            ),

                        "People":
                            scene.get(
                                "maximum_people",
                                0,
                            ),
                    }
                )

            if timeline_rows:

                st.dataframe(
                    timeline_rows,
                    width="stretch",
                    hide_index=True,
                )

            # ==================================================
            # FINISHED
            # ==================================================

            progress.progress(100)

            status.success(
                "✅ Video analysis completed!"
            )

        # ======================================================
        # ERROR HANDLING
        # ======================================================

        except Exception as error:

            progress.progress(100)

            status.error(
                "❌ Video analysis failed."
            )

            st.error(
                f"{type(error).__name__}: "
                f"{error}"
            )

            with st.expander(
                "🔧 Technical traceback"
            ):

                st.code(
                    traceback.format_exc()
                )

            st.info(
                "If this error appears again, "
                "send me the complete traceback and "
                "we'll fix the exact module causing it."
            )

    # ============================================================
    # TIME-RANGE QUESTIONS
    # ============================================================

    analyzed_video = st.session_state.get(
        "video_analysis",
        [],
    )

    if analyzed_video:

        st.markdown("---")
        st.subheader("🔎 Ask what happens in a time range")
        st.caption(
            "Enter the range as minutes and seconds. "
            "For example, 10:30 means 10 minutes and 30 seconds."
        )

        start_columns = st.columns(2)
        with start_columns[0]:
            start_minutes = st.number_input(
                "Start minutes",
                min_value=0,
                value=0,
                step=1,
                key="query_start_minutes",
            )
        with start_columns[1]:
            start_seconds = st.number_input(
                "Start seconds",
                min_value=0.0,
                max_value=59.99,
                value=0.0,
                step=1.0,
                format="%.2f",
                key="query_start_seconds",
            )

        end_columns = st.columns(2)
        with end_columns[0]:
            end_minutes = st.number_input(
                "End minutes",
                min_value=0,
                value=0,
                step=1,
                key="query_end_minutes",
            )
        with end_columns[1]:
            end_seconds = st.number_input(
                "End seconds",
                min_value=0.0,
                max_value=59.99,
                value=10.0,
                step=1.0,
                format="%.2f",
                key="query_end_seconds",
            )

        query_start = (start_minutes * 60) + start_seconds
        query_end = (end_minutes * 60) + end_seconds

        if st.button("Show this time range"):
            try:
                if query_end <= query_start:
                    raise ValueError(
                        "End time must be greater than start time."
                    )
                if query_end - query_start < 0.1:
                    raise ValueError(
                        "Please select at least 0.1 seconds of video."
                    )

                range_result = describe_time_range(
                    analyzed_video,
                    query_start,
                    query_end,
                )

                st.info(range_result["description"])

                if range_result["scenes"]:
                    st.markdown("### Matching scenes")
                    for scene in range_result["scenes"]:
                        st.write(
                            f"Scene #{scene.get('scene_number', '?')} — "
                            f"{_format_timestamp(scene.get('start_time', 0.0))} → "
                            f"{_format_timestamp(scene.get('end_time', 0.0))}"
                        )

                st.markdown("### Selected video range")
                selected_clip = _extract_video_range(
                    temp_video_path,
                    query_start,
                    query_end,
                )
                st.video(
                    selected_clip,
                    width="stretch",
                )

            except (TypeError, ValueError, RuntimeError) as error:
                st.error(str(error))
