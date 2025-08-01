# YOLOv8 Object Detection - Accuracy Improvement Guide

This guide explains how to improve object detection accuracy using the enhanced YOLOv8 system.

## 🎯 Accuracy Improvement Features

### 1. **Ensemble Detection**
Combine multiple models for better accuracy:
```bash
# Use multiple models for ensemble detection
python app.py --input image.jpg --ensemble-models yolov8s.pt yolov8m.pt
```

### 2. **Test Time Augmentation (TTA)**
Apply augmentations during inference:
```bash
# Enable TTA for improved accuracy
python app.py --input image.jpg --tta
```

### 3. **Confidence Calibration**
Adjust confidence scores for better probability estimates:
```bash
# Use confidence calibration
python app.py --input image.jpg --calibration-factor 1.2
```

### 4. **Custom Training**
Train models on your specific data:
```bash
# Run training script
python train_model.py
```

## 🚀 Quick Start for Better Accuracy

### Basic Improved Detection
```bash
# Enable all accuracy improvements
python app.py --input image.jpg --tta --calibration-factor 1.2
```

### Ensemble Detection
```bash
# Use multiple models
python app.py --input image.jpg --ensemble-models yolov8s.pt yolov8m.pt --tta
```

### Webcam with Improved Accuracy
```bash
# Real-time detection with accuracy improvements
python app.py --webcam --tta --calibration-factor 1.1
```

## 📊 Training Your Own Model

### 1. **Prepare Your Dataset**
```
training/datasets/custom_dataset/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
└── labels/
    ├── train/          # Training labels (YOLO format)
    └── val/            # Validation labels (YOLO format)
```

### 2. **Label Format**
YOLO format: `class_id x_center y_center width height`
- All values are normalized (0-1)
- One line per object

### 3. **Start Training**
```bash
python train_model.py
```

### 4. **Use Trained Model**
```bash
python app.py --input image.jpg --model training/models/trained_yolov8n.pt --tta
```

## 🔧 Advanced Configuration

### Model Selection
```bash
# Different base models
python app.py --model yolov8n.pt  # Fastest
python app.py --model yolov8s.pt  # Balanced
python app.py --model yolov8m.pt  # More accurate
python app.py --model yolov8l.pt  # Very accurate
python app.py --model yolov8x.pt  # Most accurate
```

### Confidence Thresholds
```bash
# Adjust confidence threshold
python app.py --input image.jpg --conf 0.5  # Higher confidence
python app.py --input image.jpg --conf 0.1  # Lower confidence
```

### NMS Threshold
```bash
# Adjust NMS threshold
python app.py --input image.jpg --nms 0.3  # More aggressive NMS
python app.py --input image.jpg --nms 0.7  # Less aggressive NMS
```

## 📈 Performance Comparison

### Standard vs Improved Detection

| Feature | Standard | Improved | Improvement |
|---------|----------|----------|-------------|
| Single Model | ✅ | ✅ | Baseline |
| Ensemble | ❌ | ✅ | +5-15% mAP |
| TTA | ❌ | ✅ | +2-8% mAP |
| Calibration | ❌ | ✅ | +1-3% mAP |
| Custom Training | ❌ | ✅ | +10-30% mAP |

### Model Comparison

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv8n | ⚡⚡⚡ | 37.3% mAP | 6.2MB |
| YOLOv8s | ⚡⚡ | 44.9% mAP | 22.6MB |
| YOLOv8m | ⚡ | 50.2% mAP | 52.2MB |
| YOLOv8l | 🐌 | 52.9% mAP | 87.7MB |
| YOLOv8x | 🐌🐌 | 53.9% mAP | 136.2MB |

## 🎯 Best Practices

### 1. **For Speed-Critical Applications**
```bash
python app.py --input image.jpg --model yolov8n.pt --conf 0.4
```

### 2. **For Accuracy-Critical Applications**
```bash
python app.py --input image.jpg --model yolov8x.pt --ensemble-models yolov8l.pt --tta --calibration-factor 1.2
```

### 3. **For Real-Time Applications**
```bash
python app.py --webcam --model yolov8s.pt --tta
```

### 4. **For Custom Objects**
```bash
# Train on your data first
python train_model.py

# Then use trained model
python app.py --input image.jpg --model training/models/trained_yolov8n.pt --tta
```

## 🔍 Troubleshooting

### Low Detection Accuracy
1. **Increase model size**: Use YOLOv8s/m/l/x instead of YOLOv8n
2. **Enable TTA**: Add `--tta` flag
3. **Use ensemble**: Add `--ensemble-models yolov8s.pt yolov8m.pt`
4. **Train custom model**: Use `train_model.py`

### High False Positives
1. **Increase confidence threshold**: Use `--conf 0.5` or higher
2. **Adjust NMS**: Use `--nms 0.3` for more aggressive filtering
3. **Use calibration**: Add `--calibration-factor 1.2`

### Slow Performance
1. **Use smaller model**: Switch to YOLOv8n
2. **Disable TTA**: Remove `--tta` flag
3. **Use single model**: Remove ensemble models
4. **Lower image resolution**: Resize input images

## 📊 Monitoring and Evaluation

### Check Detection Quality
```bash
# Compare different settings
python app.py --input image.jpg --model yolov8n.pt --conf 0.3
python app.py --input image.jpg --model yolov8s.pt --conf 0.3 --tta
python app.py --input image.jpg --model yolov8m.pt --conf 0.3 --tta --ensemble-models yolov8s.pt
```

### Validate Trained Models
```bash
python train_model.py  # Includes validation
```

### Export Models
```python
from src.trainer import YOLOv8Trainer
trainer = YOLOv8Trainer()
trainer.export_model("training/models/trained_yolov8n.pt", "onnx")
```

## 🎯 Success Metrics

### Good Accuracy Indicators
- **mAP50 > 0.8**: Excellent for most applications
- **mAP50 > 0.6**: Good for general use
- **mAP50 > 0.4**: Acceptable for basic applications

### Performance Targets
- **Real-time**: < 100ms per frame
- **Batch processing**: < 2s per image
- **Webcam**: 15-30 FPS

## 🚀 Next Steps

1. **Start with standard detection**: `python app.py --input image.jpg`
2. **Enable TTA**: `python app.py --input image.jpg --tta`
3. **Try ensemble**: `python app.py --input image.jpg --ensemble-models yolov8s.pt --tta`
4. **Train custom model**: `python train_model.py`
5. **Use trained model**: `python app.py --input image.jpg --model training/models/trained_yolov8n.pt --tta`

---

**Happy Detecting with Improved Accuracy! 🎯** 