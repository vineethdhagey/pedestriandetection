# Placeholder
"""
Load YOLOv8 model using Ultralytics.
"""

from ultralytics import YOLO

def load_model(config):
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    model = YOLO(model_name)
    if pretrained:
        model.load(model_name)
    return model
