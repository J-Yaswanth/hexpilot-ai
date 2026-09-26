import streamlit as st
import cv2
import tempfile
import os

from ultralytics import YOLO

from vision.tracker import ObjectTracker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Video Intelligence Studio",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

TRACKER_CONFIG = "trackers/botsort_custom.yaml"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🎬 AI Video Intelligence Studio"
)

st.markdown(
    """
    ### Turn raw footage into AI-powered video intelligence.

    Upload a video and let AI understand what's inside it.
    """
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "🎥 Upload your video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


if uploaded_file is not None:

    st.success(
        "✅ Video uploaded successfully!"
    )


    # ========================================================
    # SAVE VIDEO TEMPORARILY
    # ========================================================

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        video_path = temp_file.name


    # ========================================================
    # ORIGINAL VIDEO
    # ========================================================

    st.subheader(
        "🎥 Original Video"
    )

    st.video(
        video_path
    )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )


    if cap.isOpened():

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        duration = (
            frame_count / fps
            if fps > 0
            else 0
        )


        st.subheader(
            "📊 Video Information"
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Resolution",
            f"{width} × {height}"
        )


        col2.metric(
            "FPS",
            f"{fps:.2f}"
        )


        col3.metric(
            "Frames",
            f"{int(frame_count):,}"
        )


        col4.metric(
            "Duration",
            f"{duration:.2f} sec"
        )


        cap.release()


    # ========================================================
    # ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 AI Video Analysis"
    )


    analyze_button = st.button(
        "🚀 Analyze Video",
        type="primary"
    )


    if analyze_button:

        st.info(
            "🧠 AI is analyzing the video "
            "frame-by-frame..."
        )


        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(
            video_path
        )


        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            st.stop()


        # ====================================================
        # TOTAL FRAMES
        # ====================================================

        total_video_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # ====================================================
        # RESULTS
        # ====================================================

        frame_detection_counts = {}

        tracker = ObjectTracker()


        processed_frames = 0


        progress_bar = st.progress(
            0
        )


        # ====================================================
        # PROCESS EVERY FRAME
        # ====================================================

        while True:

            ret, frame = cap.read()


            if not ret:

                break


            processed_frames += 1


            # =================================================
            # YOLO + BOT-SORT + REID
            # =================================================

            results = model.track(

                frame,

                conf=CONFIDENCE_THRESHOLD,

                persist=True,

                tracker=TRACKER_CONFIG,

                verbose=False
            )


            # =================================================
            # DETECTIONS
            # =================================================

            frame_detections = []


            for result in results:

                if result.boxes is None:

                    continue


                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )


                    confidence = float(
                        box.conf[0]
                    )


                    class_name = model.names[
                        class_id
                    ]


                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )


                    track_id = None


                    if box.id is not None:

                        track_id = int(
                            box.id[0]
                        )


                    detection = {

                        "label": class_name,

                        "confidence": confidence,

                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],

                        "track_id": track_id
                    }


                    frame_detections.append(
                        detection
                    )


                    # =================================================
                    # FRAME DETECTION COUNT
                    # =================================================

                    if class_name not in frame_detection_counts:

                        frame_detection_counts[
                            class_name
                        ] = 0


                    frame_detection_counts[
                        class_name
                    ] += 1


            # =================================================
            # UPDATE OUR TRACKER
            # =================================================

            tracker.update(
                frame_detections
            )


            # =================================================
            # PROGRESS
            # =================================================

            if total_video_frames > 0:

                progress = min(
                    int(
                        (
                            processed_frames
                            / total_video_frames
                        ) * 100
                    ),
                    100
                )

                progress_bar.progress(
                    progress
                )


        # ====================================================
        # FINISH
        # ====================================================

        cap.release()

        progress_bar.progress(
            100
        )


        st.success(
            "🎉 Video analysis completed!"
        )


        # ====================================================
        # FRAME DETECTIONS
        # ====================================================

        st.subheader(
            "🔍 Frame Detection Results"
        )


        st.caption(
            "A frame detection means an object was "
            "recognized in one analyzed frame. "
            "This is NOT the number of physical objects."
        )


        if frame_detection_counts:

            columns = st.columns(
                min(
                    len(frame_detection_counts),
                    4
                )
            )


            for index, (
                object_name,
                count
            ) in enumerate(
                sorted(
                    frame_detection_counts.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            ):

                columns[
                    index % len(columns)
                ].metric(
                    object_name.title(),
                    count
                )


        else:

            st.warning(
                "No objects were detected."
            )


        # ====================================================
        # UNIQUE TRACKS
        # ====================================================

        st.subheader(
            "🎯 Unique Tracked Objects"
        )


        st.caption(
            "These are unique tracking identities created "
            "by the tracking model. They should not be "
            "interpreted as a guaranteed real-world count."
        )


        unique_counts = (
            tracker.get_unique_counts()
        )


        if unique_counts:

            columns = st.columns(
                min(
                    len(unique_counts),
                    4
                )
            )


            for index, (
                object_name,
                count
            ) in enumerate(
                sorted(
                    unique_counts.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            ):

                columns[
                    index % len(columns)
                ].metric(
                    object_name.title(),
                    count
                )

        else:

            st.info(
                "No persistent tracks were generated."
            )


        # ====================================================
        # TRACK DETAILS
        # ====================================================

        st.subheader(
            "🧾 Track Details"
        )


        all_tracks = (
            tracker.get_all_tracks()
        )


        if all_tracks:

            for track in sorted(
                all_tracks,
                key=lambda item: (
                    item["label"],
                    item["track_id"]
                )
            ):

                st.write(
                    f"**{track['label'].title()} "
                    f"#{track['track_id']}** — "
                    f"confidence "
                    f"{track['confidence']:.2f}"
                )

        else:

            st.write(
                "No track details available."
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        st.subheader(
            "📋 AI Analysis Summary"
        )


        if unique_counts:

            for (
                object_name,
                unique_count
            ) in sorted(
                unique_counts.items(),
                key=lambda item: item[1],
                reverse=True
            ):

                frame_count_for_object = (
                    frame_detection_counts.get(
                        object_name,
                        0
                    )
                )


                st.write(
                    f"**{object_name.title()}** — "
                    f"{unique_count} tracking identity(ies) | "
                    f"{frame_count_for_object} frame detections"
                )

        else:

            st.write(
                "No objects were successfully tracked."
            )


        # ====================================================
        # PROJECT STATUS
        # ====================================================

        st.info(
            "🚧 Next: reliable object inventory, "
            "timestamps, person attributes, "
            "person-object relationships, activities, "
            "scene understanding, audio intelligence, "
            "video Q&A and AI-generated summaries."
        )