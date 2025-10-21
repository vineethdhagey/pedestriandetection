# Placeholder
"""
Helper functions and Streamlit dashboard.
"""

import streamlit as st
import cv2
import os

def launch_dashboard():
    st.title("Multi-View Object Detection Dashboard")
    image_dir = "data/processed/cam1"
    images = sorted(os.listdir(image_dir))[:10]

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        st.image(cv2.imread(img_path), caption=img_name)
