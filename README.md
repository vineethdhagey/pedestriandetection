 # 🧠Multi-View Object Detection Pipeline

This project implements a multi-camera object detection pipeline using YOLOv8 and PyTorch. It supports preprocessing, training, evaluation, and fusion of detections across views

## Features

📹 Multi-camera video preprocessing (WildTrack format)

🧠 YOLOv8-based pedestrian detection

🔁 Config-driven training and inference

📊 Evaluation metrics: mAP, precision, recall

🖼 Dummy outputs for demo and resume use

🖥 Streamlit dashboard for visualizing predictions

🧩 Modular CLI interface via main.py


🧪 Project Structure
```

multi_view_object_detection/
├── data/                      # Raw and processed WildTrack data
├── logs/                      # Training logs and model checkpoints
├── runs/detect/predict/      # Inference outputs (dummy or real)
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── infer.py
│   ├── evaluate.py
│   ├── dashboard.py
│   └── utils.py
├── main.py                   # CLI task router
├── requirements.txt
└── README.md

```

## 🚀 Getting Started
```bash
git clone https://github.com/vineethdhagey/pedestriandetection.git
cd pedestrian detection
pip install -r requirements.txt
```
## 🧭 Usage
🔧 Preprocess WildTrack Data
```bash
python main.py --task preprocess
```

## 🧠 Train YOLOv8 Model (optional)
```bash
python main.py --task train
```
🔍 Run Inference (dummy or trained model)
```bash
python main.py --task infer
```
📊 Evaluate Model Performance
```bash
python main.py --task evaluate
```


🖥 Launch Streamlit Dashboard
```bash
streamlit run main.py -- --task dashboard
```

