# YOLOv11 Object Detection System

A comprehensive object detection system using YOLOv11 with advanced accuracy improvement features.

## 🌟 Features

### Object Detection
- **Real-time detection** using webcam
- **Image and video processing** with bounding boxes
- **Batch processing** for multiple files
- **Multiple model support** (YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, YOLOv11x)
- **Ensemble detection** for improved accuracy
- **Test Time Augmentation (TTA)** for better results
- **Confidence calibration** for accurate probability estimates

### Accuracy Improvements
- **Ensemble models** for better detection accuracy
- **Test Time Augmentation** for improved robustness
- **Confidence calibration** for better probability estimates
- **Custom model training** capabilities

## 📁 Project Structure

```
yolov11_object_detection/
├── data/                 # Input images/videos
├── outputs/              # Output images/videos with detections
├── models/               # Custom or downloaded YOLOv11 models
├── src/
│   ├── detector.py       # Object detection logic using YOLOv11
│   ├── config.py         # Configurations (thresholds, model path, etc.)
│   └── utils.py          # Helper functions (draw boxes, load files)
├── app.py                # Main script to run detection
├── requirements.txt      # List of dependencies
└── README.md             # Documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)

### Install Dependencies

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

### First Run

On first run, the system will automatically download the YOLOv11n model if not present:

```bash
python app.py --help
```

## 🚀 Quick Start

### Basic Object Detection
```bash
# Detect objects in an image
python app.py --input data/image.jpg --output outputs/detected.jpg

# Real-time webcam detection
python app.py --webcam

# Process video file
python app.py --input data/video.mp4 --output outputs/detected_video.mp4
```
