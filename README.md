# 🧠 Multi-View Object Detection Pipeline

A comprehensive pipeline for multi-camera pedestrian detection using YOLOv8 and PyTorch. This project processes WildTrack dataset videos, trains object detection models, performs inference across multiple views, evaluates performance, and visualizes results through a Streamlit dashboard.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **📹 Multi-Camera Video Preprocessing**: Extract frames from WildTrack format videos for each camera view
- **🧠 YOLOv8 Object Detection**: Train and deploy YOLOv8 models for pedestrian detection
- **🔄 Config-Driven Pipeline**: Flexible configuration via YAML files for easy experimentation
- **📊 Performance Evaluation**: Compute mAP, precision, recall, and other metrics
- **🔗 Detection Fusion**: Combine detections across multiple camera views using non-maximum suppression
- **🖥️ Interactive Dashboard**: Visualize predictions and results with Streamlit
- **🧩 Modular CLI**: Simple command-line interface for all pipeline tasks

## 🏗️ Project Structure

```
multi_view_object_detection/
├── main.py                              # CLI entry point for running tasks
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── yolov8n.pt                           # Pre-trained YOLOv8 nano model
├── yolo11n.pt                           # Pre-trained YOLO11 nano model
├── configs/                             # Configuration files
│   ├── model_config.yaml                # Model and training settings
│   └── dataset_config.yaml              # Dataset paths and parameters
├── data/                                # Data directories
│   ├── raw/                             # Raw WildTrack dataset
│   │   └── wildtrack.v1i.yolov8/        # YOLOv8 formatted dataset
│   └── processed/                       # Extracted frames per camera
│       ├── cam1/
│       ├── cam2/
│       └── cam3/
├── logs/                                # Training logs and checkpoints
│   └── wildtrack_yolov8/                # YOLOv8 training outputs
├── notebooks/                           # Jupyter notebooks for exploration
│   └── test.ipynb
└── src/                                # Source code modules
    ├── data_preprocessing.py            # Video frame extraction
    ├── dataset.py                       # Custom PyTorch dataset class
    ├── model.py                         # Model loading utilities
    ├── train.py                         # Training script
    ├── infer.py                         # Inference and prediction
    ├── evaluate.py                      # Model evaluation metrics
    ├── fuse_detections.py               # Multi-view detection fusion
    ├── dashboard.py                     # Streamlit visualization
    └── utils.py                         # Helper functions
```

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vineethdhagey/pedestriandetection.git
   cd multi_view_object_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset** (if not present):
   - The WildTrack dataset should be placed in `data/raw/wildtrack.v1i.yolov8/`
   - Ensure the dataset is in YOLOv8 format with `data.yaml`, train/val/test splits

## 🧭 Usage

The pipeline is controlled via the `main.py` script with task-specific arguments.

### 🔧 Preprocess Data
Extract frames from multi-camera videos:
```bash
python main.py --task preprocess
```

### 🧠 Train Model
Train YOLOv8 on the WildTrack dataset:
```bash
python main.py --task train
```

### 🔍 Run Inference
Perform object detection on test images:
```bash
python main.py --task infer
```

### 📊 Evaluate Performance
Compute evaluation metrics on predictions:
```bash
python main.py --task evaluate
```

### 🖥️ Launch Dashboard
Visualize results in a web interface:
```bash
streamlit run main.py -- --task dashboard
```

## ⚙️ Configuration

Customize the pipeline using YAML config files in the `configs/` directory:

- **`model_config.yaml`**: Model architecture, training hyperparameters, and paths
- **`dataset_config.yaml`**: Dataset locations, camera configurations, and preprocessing settings

Example `model_config.yaml`:
```yaml
model:
  name: yolov8n
  pretrained: true
  input_size: 640
  num_classes: 1  # Pedestrian class

training:
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  checkpoint_dir: logs/checkpoints
  log_dir: logs/tensorboard
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



*Built using PyTorch, Ultralytics YOLOv8, and Streamlit*

![Dashboard Screenshot](https://github.com/user-attachments/assets/0e9a0455-c818-4a06-9fbc-232fae853762)
