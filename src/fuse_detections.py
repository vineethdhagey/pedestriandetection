# src/fuse_detections.py
"""
Fuse detections across multiple views using calibration and NMS.
"""

import os
import numpy as np

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    Each box is [x1, y1, x2, y2].
    """
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Apply NMS across detections.
    detections: list of dicts with keys {"bbox": [x1,y1,x2,y2], "conf": float, "cam": str}
    """
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [d for d in detections if compute_iou(best["bbox"], d["bbox"]) < iou_threshold]
    return keep

def load_yolo_predictions(pred_dir="runs/detect/predict2/labels"):
    detections = []
    if not os.path.exists(pred_dir):
        print(f"Prediction directory {pred_dir} not found.")
        return detections

    for file in os.listdir(pred_dir):
        if file.endswith(".txt"):
            cam_id = os.path.basename(pred_dir)
            with open(os.path.join(pred_dir, file)) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:  # no confidence
                        cls, xc, yc, w, h = map(float, parts)
                        conf = 1.0
                    elif len(parts) == 6:  # with confidence
                        cls, xc, yc, w, h, conf = map(float, parts)
                    else:
                        continue
                    x1 = xc - w/2
                    y1 = yc - h/2
                    x2 = xc + w/2
                    y2 = yc + h/2
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "cam": cam_id
                    })
    return detections

import json

def run():
    """
    Entry point for fuse task.
    Loads YOLO inference outputs, applies NMS, prints fused detections,
    and saves them to runs/fuse/fused.json.
    """
    # automatically pick the latest predict folder
    base_dir = "runs/detect"
    if not os.path.exists(base_dir):
        print("No prediction outputs found.")
        return
    preds = [d for d in os.listdir(base_dir) if d.startswith("predict")]
    if not preds:
        print("No prediction outputs found.")
        return
    preds.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
    pred_dir = os.path.join(base_dir, preds[0], "labels")

    detections = load_yolo_predictions(pred_dir)
    fused = non_max_suppression(detections, iou_threshold=0.5)

    print(f"Using predictions from: {pred_dir}")
    print("Fused detections:", fused)

    # --- Save fused detections ---
    os.makedirs("runs/fuse", exist_ok=True)
    out_file = "runs/fuse/fused.json"
    with open(out_file, "w") as f:
        json.dump(fused, f, indent=2)
    print(f"Fused detections saved to {out_file}")

