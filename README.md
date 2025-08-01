# YOLOv8 Object Detection System

A modular and comprehensive object detection system built with YOLOv8, supporting image, video, and webcam detection with customizable configurations.

## 🚀 Features

- **Multi-format Support**: Process images (JPG, PNG, BMP, TIFF, WebP) and videos (MP4, AVI, MOV, MKV, WMV, FLV)
- **Real-time Detection**: Webcam support for live object detection
- **Batch Processing**: Process entire directories of images/videos
- **Customizable**: Adjustable confidence thresholds, NMS settings, and model selection
- **Modular Design**: Clean separation of concerns with configurable components
- **CLI Interface**: Easy-to-use command-line interface with comprehensive options

## 📁 Project Structure

```
yolov8_object_detection/
├── data/                 # Input images/videos
├── outputs/              # Output images/videos with detections
├── models/               # Custom or downloaded YOLOv8 models
├── src/
│   ├── detector.py       # Object detection logic using YOLOv8
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

On first run, the system will automatically download the YOLOv8n model if not present:

```bash
python app.py --help
```

## 🎯 Usage

### Basic Examples

#### 1. Single Image Detection

```bash
# Detect objects in a single image
python app.py --input data/image.jpg --output outputs/detected_image.jpg

# Show result in window
python app.py --input data/image.jpg --output outputs/detected_image.jpg --show
```

#### 2. Video Processing

```bash
# Process video file
python app.py --input data/video.mp4 --output outputs/detected_video.mp4

# Process video with real-time display
python app.py --input data/video.mp4 --output outputs/detected_video.mp4 --show
```

#### 3. Batch Processing

```bash
# Process all images in a directory
python app.py --input data/ --output outputs/ --batch

# Process with custom confidence threshold
python app.py --input data/ --output outputs/ --batch --conf 0.5
```

#### 4. Webcam Detection

```bash
# Real-time webcam detection
python app.py --webcam

# Save webcam output to video
python app.py --webcam --output outputs/webcam_recording.mp4

# Use different webcam device
python app.py --webcam --webcam-index 1
```

### Advanced Options

#### Custom Model

```bash
# Use a different YOLOv8 model
python app.py --input data/image.jpg --model models/yolov8s.pt

# Use custom trained model
python app.py --input data/image.jpg --model models/custom_model.pt
```

#### Detection Settings

```bash
# Adjust confidence threshold (0.0 to 1.0)
python app.py --input data/image.jpg --conf 0.7

# Adjust NMS threshold
python app.py --input data/image.jpg --nms 0.3

# Combine multiple settings
python app.py --input data/image.jpg --conf 0.5 --nms 0.4 --model models/yolov8m.pt
```

## 📊 Supported Formats

### Input Formats

**Images:**
- JPG, JPEG
- PNG
- BMP
- TIFF
- WebP

**Videos:**
- MP4
- AVI
- MOV
- MKV
- WMV
- FLV

### Output Formats

- Images: Same as input format
- Videos: MP4 (H.264 codec)

## ⚙️ Configuration

The system uses a centralized configuration file (`src/config.py`) with the following key settings:

### Detection Parameters

```python
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
NMS_THRESHOLD = 0.45         # Non-maximum suppression threshold
MAX_DETECTIONS = 300          # Maximum number of detections
```

### Display Settings

```python
BOX_THICKNESS = 2             # Bounding box line thickness
TEXT_THICKNESS = 1            # Text line thickness
TEXT_SCALE = 0.6             # Text size scale
```

### Model Settings

```python
DEFAULT_MODEL = "yolov8n.pt" # Default YOLOv8 model
```

## 🔧 Available Models

The system supports all YOLOv8 variants:

- **YOLOv8n**: Nano (fastest, smallest)
- **YOLOv8s**: Small
- **YOLOv8m**: Medium
- **YOLOv8l**: Large
- **YOLOv8x**: Extra Large (most accurate)

Download models automatically or place custom models in the `models/` directory.

## 📈 Performance Tips

1. **GPU Acceleration**: Install CUDA for faster inference
2. **Model Selection**: Use smaller models (YOLOv8n) for speed, larger models for accuracy
3. **Confidence Threshold**: Higher thresholds reduce false positives but may miss objects
4. **Batch Processing**: Process multiple images for better throughput

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Failed**
   ```bash
   # Manual download
   pip install ultralytics
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Webcam Not Working**
   ```bash
   # Try different webcam index
   python app.py --webcam --webcam-index 1
   ```

4. **Memory Issues**
   - Reduce image/video resolution
   - Use smaller model (YOLOv8n)
   - Lower confidence threshold

### Error Messages

- `"Failed to load image"`: Check file path and format
- `"Failed to open video file"`: Verify video codec support
- `"Could not open webcam"`: Check webcam permissions and index

## 🔄 API Usage

The system can also be used programmatically:

```python
from src.detector import YOLOv8Detector

# Initialize detector
detector = YOLOv8Detector(
    model_path="yolov8n.pt",
    confidence_threshold=0.25
)

# Detect in image
result = detector.detect_image("data/image.jpg", "outputs/result.jpg")

# Process video
result = detector.detect_video("data/video.mp4", "outputs/result.mp4")

# Webcam detection
detector.detect_webcam(output_path="outputs/webcam.mp4")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [OpenCV](https://opencv.org/) for computer vision operations
- [PyTorch](https://pytorch.org/) for deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Happy Detecting! 🎯** # Object-Detection-System
# Object-Detection-System
# Object-Detection-System
# Object-Detection-System
