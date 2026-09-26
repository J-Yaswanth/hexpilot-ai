from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from training.common import (
    ANNOTATIONS_CSV,
    ANNOTATION_FIELDS,
    DEFAULT_VIDEO_DIR,
    SCENE_CATEGORIES,
    atomic_write_csv,
    capture_scene_preview,
    read_csv_rows,
)

st.set_page_config(page_title="Scene Labeling", page_icon="🎞️", layout="wide")
st.title("Scene segment labeling")
st.write(
    "Review the preview frames and assign a label only when you can confirm it. "
    "Labels are never generated automatically."
)

video_dir = Path(st.sidebar.text_input("Video folder", str(DEFAULT_VIDEO_DIR)))
annotations_path = Path(st.sidebar.text_input("Annotations CSV", str(ANNOTATIONS_CSV)))

if not annotations_path.is_file():
    st.error(f"Annotations file not found: {annotations_path}. Run dataset preparation first.")
    st.stop()

rows = read_csv_rows(annotations_path)
if not rows:
    st.warning("There are no scene segments to label. Run dataset preparation first.")
    st.stop()

for row in rows:
    row.setdefault("scene_id", "")
    row.setdefault("scene_label", "")

pending = [
    index for index, row in enumerate(rows)
    if row.get("scene_label", "").strip().casefold() not in {
        category.casefold() for category in SCENE_CATEGORIES
    }
]
st.caption(
    f"{len(rows) - len(pending)} of {len(rows)} segments labeled; "
    f"{len(pending)} still need review."
)

def scene_option(index: int) -> str:
    row = rows[index]
    return (
        f"{row['video']} | {float(row['start_time']):.2f}s–"
        f"{float(row['end_time']):.2f}s"
    )

options = ["— Unlabeled —", *SCENE_CATEGORIES]

selected_index = st.selectbox(
    "Choose a detected scene",
    range(len(rows)),
    format_func=scene_option,
    key="selected_scene_index",
)
row = rows[selected_index]
current_label = row.get("scene_label", "").strip()
scene_identity = row.get("scene_id") or scene_option(selected_index)
if st.session_state.get("_label_scene_identity") != scene_identity:
    st.session_state["label_choice"] = (
        current_label if current_label in SCENE_CATEGORIES else options[0]
    )
    st.session_state["_label_scene_identity"] = scene_identity
video_path = video_dir / row["video"]
start_time = float(row["start_time"])
end_time = float(row["end_time"])
st.subheader(scene_option(selected_index))

if not video_path.is_file():
    st.error(f"Video file not found: {video_path}")
    st.stop()

try:
    frames = capture_scene_preview(video_path, start_time, end_time)
except (OSError, RuntimeError, ValueError) as exc:
    st.error(f"Could not preview this segment: {exc}")
    st.stop()

if frames:
    columns = st.columns(len(frames))
    for index, (column, frame) in enumerate(zip(columns, frames), start=1):
        column.image(frame, caption=f"Preview {index}", use_container_width=True)
else:
    st.warning("No preview frames could be read for this segment.")

selected_label = st.selectbox(
    "Confirmed scene category",
    options,
    key="label_choice",
)
st.caption("Use “other” only when the scene is understandable but does not fit the available categories.")

if st.button("Save label", type="primary"):
    if selected_label == "— Unlabeled —":
        row["scene_label"] = ""
    else:
        row["scene_label"] = selected_label
    atomic_write_csv(annotations_path, ANNOTATION_FIELDS, rows)
    st.success(f"Saved {selected_label if selected_label != '— Unlabeled —' else 'blank'} label.")
    st.rerun()
