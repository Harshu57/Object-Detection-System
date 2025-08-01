#!/usr/bin/env python3
"""
Test script for YOLOv8 Object Detection System
"""

import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        from src import config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        from src import utils
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Utils import failed: {e}")
        return False
    
    try:
        from src.detector import YOLOv8Detector
        print("✓ Detector module imported successfully")
    except ImportError as e:
        print(f"✗ Detector import failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration settings."""
    print("\nTesting configuration...")
    
    try:
        from src import config
        
        # Test directory creation
        assert config.DATA_DIR.exists() or config.DATA_DIR.mkdir(exist_ok=True)
        assert config.OUTPUT_DIR.exists() or config.OUTPUT_DIR.mkdir(exist_ok=True)
        assert config.MODELS_DIR.exists() or config.MODELS_DIR.mkdir(exist_ok=True)
        print("✓ Directories created successfully")
        
        # Test configuration values
        assert 0 <= config.CONFIDENCE_THRESHOLD <= 1
        assert 0 <= config.NMS_THRESHOLD <= 1
        assert config.MAX_DETECTIONS > 0
        print("✓ Configuration values are valid")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from src import utils
        import numpy as np
        
        # Test color function
        color = utils.get_color_for_class(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
        print("✓ Color utility works")
        
        # Test time formatting
        time_str = utils.format_time(3661)  # 1 hour, 1 minute, 1 second
        assert time_str == "01:01:01"
        print("✓ Time formatting works")
        
        return True
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def test_detector_initialization():
    """Test detector initialization."""
    print("\nTesting detector initialization...")
    
    try:
        from src.detector import YOLOv8Detector
        
        # Test detector initialization
        detector = YOLOv8Detector(
            confidence_threshold=0.25,
            nms_threshold=0.45
        )
        
        # Check if model is loaded
        assert detector.model is not None
        assert hasattr(detector.model, 'names')
        print("✓ Detector initialized successfully")
        print(f"✓ Model loaded with {len(detector.class_names)} classes")
        
        return True
    except Exception as e:
        print(f"✗ Detector initialization failed: {e}")
        return False


def test_model_download():
    """Test if YOLOv8 model can be downloaded."""
    print("\nTesting model download...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load/download the default model
        model = YOLO("yolov8n.pt")
        print("✓ YOLOv8n model loaded/downloaded successfully")
        
        # Test basic inference
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, conf=0.25)
        print("✓ Model inference test passed")
        
        return True
    except Exception as e:
        print(f"✗ Model download/inference failed: {e}")
        return False


def test_cli_help():
    """Test if CLI help works."""
    print("\nTesting CLI help...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "app.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✓ CLI help works")
            return True
        else:
            print(f"✗ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("YOLOv8 Object Detection System - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Utils Test", test_utils),
        ("Detector Initialization Test", test_detector_initialization),
        ("Model Download Test", test_model_download),
        ("CLI Help Test", test_cli_help),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place images/videos in the 'data/' directory")
        print("2. Run: python app.py --input data/image.jpg --output outputs/result.jpg")
        print("3. Or run: python example.py for examples")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check internet connection for model download")
        print("3. Ensure you have Python 3.8+ installed")


if __name__ == "__main__":
    main() 