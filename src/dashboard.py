import streamlit as st
import os
from PIL import Image

def run():
    st.title("YOLOv8 Predictions Viewer")

    result_dir = "runs/detect/predict"
    if not os.path.exists(result_dir):
        st.warning("No predictions found. Run inference first.")
        return

    images = [f for f in os.listdir(result_dir) if f.endswith(".jpg")]
    for img_file in images:
        img_path = os.path.join(result_dir, img_file)
        st.image(Image.open(img_path), caption=img_file, use_column_width=True)
