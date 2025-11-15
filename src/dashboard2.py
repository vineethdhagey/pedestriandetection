import streamlit as st
import os
from PIL import Image

CAMERAS = ["C1","C2","C3","C4","C5","C6","C7"]

def get_latest_predict_dir(base="runs/detect"):
    if not os.path.exists(base):
        return None
    preds = [d for d in os.listdir(base) if d.startswith("predict")]
    if not preds:
        return None
    preds.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)), reverse=True)
    return os.path.join(base, preds[0])

def extract_frame_ids(images):
    frame_ids = set()
    for img in images:
        parts = img.split("_")
        if len(parts) > 1 and parts[0].startswith("C"):
            frame_ids.add(parts[1])  # second part is the frame ID
    return sorted(frame_ids)

def find_image(images, cam, frame_id):
    for img in images:
        # robust match: camera + frame ID anywhere in filename
        if f"{cam}_{frame_id}" in img and img.endswith(".jpg"):
            return img
    return None

def run():
    st.title("Multi‑View YOLOv8 Predictions Dashboard")

    result_dir = get_latest_predict_dir()
    if result_dir is None or not os.path.exists(result_dir):
        st.warning("No predictions found. Run inference first.")
        return

    images = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]
    frame_ids = extract_frame_ids(images)
    if not frame_ids:
        st.warning("No frame IDs found in prediction images.")
        return

    selected_frame = st.selectbox("Select a frame ID", frame_ids)

    st.subheader(f"Camera Views for Frame {selected_frame}")
    # Use fewer columns so each image is bigger
    cols = st.columns(2)
    for i, cam in enumerate(CAMERAS):
        img_name = find_image(images, cam, selected_frame)
        if img_name:
            img_path = os.path.join(result_dir, img_name)
            with cols[i % 2]:
                st.image(Image.open(img_path), caption=cam, use_container_width=True)
        else:
            with cols[i % 2]:
                st.info(f"{cam} image not found")

if __name__ == "__main__":
    run()
