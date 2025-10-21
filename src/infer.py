from ultralytics import YOLO

def run():
    model = YOLO("logs/checkpoints/yolov8_wildtrack.pt")  # Hypothetical trained model
    results = model.predict(
        source="data/raw/wildtrack.v1i.yolov8/test/images",
        save=True,
        conf=0.25
    )
    print("Inference complete. Results saved in:", results[0].save_dir)
