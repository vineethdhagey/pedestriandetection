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
├── data/                         # Raw and processed WildTrack data
│   ├── raw/    
│   └── processed/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── fuse_detections.py
│   └── utils.py
├── configs/
│   ├── model_config.yaml
│   └── dataset_config.yaml
├── logs/                                 # Training logs and model checkpoints
├── requirements.txt
├── README.md
└── main.py                              # CLI task router



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

<img width="1473" height="830" alt="a" src="https://github.com/user-attachments/assets/0e9a0455-c818-4a06-9fbc-232fae853762" />
