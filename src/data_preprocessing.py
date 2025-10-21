# Placeholder
"""
Extract frames and annotations from multi-camera videos.
"""

import os
import cv2
import yaml

def run():
    with open("configs/dataset_config.yaml") as f:
        config = yaml.safe_load(f)

    raw_dir = config['dataset']['raw_dir']
    processed_dir = config['dataset']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)

    for cam in config['dataset']['cameras']:
        cam_path = os.path.join(raw_dir, f"{cam}.mp4")
        cap = cv2.VideoCapture(cam_path)
        frame_id = 0
        cam_out_dir = os.path.join(processed_dir, cam)
        os.makedirs(cam_out_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(cam_out_dir, f"{frame_id:05d}.jpg"), frame)
            frame_id += 1

        cap.release()
        print(f"Extracted {frame_id} frames from {cam}")
