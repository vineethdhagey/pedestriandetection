# src/evaluate.py
from ultralytics import YOLO
import numpy as np

def run():
    weights_path = "logs/wildtrack_yolov8/weights/best.pt"
    data_config = "configs/data.yaml"

    model = YOLO(weights_path)
    metrics = model.val(data=data_config)

    # Convert arrays to scalars
    precision = float(np.mean(metrics.box.p))
    recall    = float(np.mean(metrics.box.r))
    map50     = float(np.mean(metrics.box.map50))
    map5095   = float(np.mean(metrics.box.map))

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"mAP@50:    {map50:.3f}")
    print(f"mAP@50-95: {map5095:.3f}")
