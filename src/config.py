"""
Configuration settings for YOLOv8 Object Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration
DEFAULT_MODEL = "yolov8n.pt"  # Default YOLOv8 nano model
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL

# Detection settings
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detection
NMS_THRESHOLD = 0.45  # Non-maximum suppression threshold
MAX_DETECTIONS = 300  # Maximum number of detections

# Input/Output settings
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']

# Display settings
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SCALE = 0.6
TEXT_COLOR = (255, 255, 255)  # White text
BOX_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Dark Red
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Blue
    (128, 128, 0),  # Olive
]

# Video processing settings
VIDEO_FPS = 30
VIDEO_FOURCC = 'mp4v'  # Video codec for output

# Webcam settings
WEBCAM_INDEX = 0  # Default webcam index
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480 