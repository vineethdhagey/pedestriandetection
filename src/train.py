"""
Train YOLOv8 model using Roboflow WildTrack dataset.
"""

from ultralytics import YOLO

def run():
    # Load a pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")  # You can change to yolov8s.pt or yolov8m.pt

    # Train using the Roboflow dataset
    model.train(
        data="data/raw/wildtrack.v1i.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        project="logs",
        name="wildtrack_yolov8",
        exist_ok=True
    )

    # Save final model
    model.save("logs/checkpoints/yolov8_wildtrack.pt")
