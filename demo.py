#!/usr/bin/env python3
"""
Demo script for YOLOv8 Object Detection System
This script demonstrates basic usage of the system.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLOv8Detector
from src import config


def demo_basic_usage():
    """Demonstrate basic usage of the YOLOv8 detector."""
    print("YOLOv8 Object Detection System - Demo")
    print("=" * 50)
    
    # Initialize detector
    print("Initializing YOLOv8 detector...")
    detector = YOLOv8Detector(
        confidence_threshold=0.3,
        nms_threshold=0.45
    )
    
    print(f"✓ Detector initialized with {len(detector.class_names)} classes")
    print(f"✓ Confidence threshold: {detector.confidence_threshold}")
    print(f"✓ NMS threshold: {detector.nms_threshold}")
    
    # Create a simple test image (you can replace this with a real image)
    import numpy as np
    import cv2
    
    # Create a test image with some shapes
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some shapes to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(test_image, (400, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.rectangle(test_image, (300, 300), (400, 400), (255, 0, 0), -1)  # Blue rectangle
    
    # Save test image
    test_image_path = "data/demo_test_image.jpg"
    Path("data").mkdir(exist_ok=True)
    cv2.imwrite(test_image_path, test_image)
    print(f"✓ Created test image: {test_image_path}")
    
    # Perform detection
    print("\nPerforming object detection...")
    result = detector.detect_image(
        test_image_path,
        "outputs/demo_result.jpg",
        show_result=False
    )
    
    if "error" not in result:
        print("✓ Detection completed successfully!")
        print(f"  - Total detections: {result['total_detections']}")
        print(f"  - Processing time: {result['processing_time']:.2f} seconds")
        print(f"  - Image shape: {result['image_shape']}")
        
        # Show detection details
        if result['detections']:
            print("\nDetections found:")
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. {detection['class_name']} "
                      f"(confidence: {detection['confidence']:.2f})")
        else:
            print("\nNo objects detected in the test image.")
            print("This is normal for a simple test image.")
        
        print(f"\n✓ Result saved to: outputs/demo_result.jpg")
    else:
        print(f"❌ Detection failed: {result['error']}")
    
    return result


def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("\n" + "=" * 50)
    print("CLI Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Single Image Detection", 
         "python app.py --input data/image.jpg --output outputs/result.jpg"),
        
        ("Image with Display", 
         "python app.py --input data/image.jpg --output outputs/result.jpg --show"),
        
        ("Video Processing", 
         "python app.py --input data/video.mp4 --output outputs/result.mp4"),
        
        ("Batch Processing", 
         "python app.py --input data/ --output outputs/ --batch"),
        
        ("Webcam Detection", 
         "python app.py --webcam"),
        
        ("Custom Confidence", 
         "python app.py --input data/image.jpg --conf 0.5"),
        
        ("Different Model", 
         "python app.py --input data/image.jpg --model yolov8s.pt"),
        
        ("Help", 
         "python app.py --help")
    ]
    
    for title, command in examples:
        print(f"\n{title}:")
        print(f"  {command}")


def demo_api_usage():
    """Demonstrate API usage."""
    print("\n" + "=" * 50)
    print("API Usage Examples:")
    print("=" * 50)
    
    code_examples = [
        ("Basic Initialization", """
from src.detector import YOLOv8Detector

detector = YOLOv8Detector(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    nms_threshold=0.45
)
"""),
        
        ("Image Detection", """
result = detector.detect_image(
    "data/image.jpg",
    "outputs/result.jpg",
    show_result=False
)
"""),
        
        ("Video Processing", """
result = detector.detect_video(
    "data/video.mp4",
    "outputs/result.mp4",
    show_result=False
)
"""),
        
        ("Webcam Detection", """
detector.detect_webcam(
    webcam_index=0,
    output_path="outputs/webcam.mp4"
)
"""),
        
        ("Batch Processing", """
result = detector.batch_detect_images(
    "data/",
    "outputs/"
)
""")
    ]
    
    for title, code in code_examples:
        print(f"\n{title}:")
        print(code)


def main():
    """Run the demo."""
    try:
        # Run basic demo
        demo_basic_usage()
        
        # Show CLI examples
        demo_cli_usage()
        
        # Show API examples
        demo_api_usage()
        
        print("\n" + "=" * 50)
        print("Demo completed!")
        print("\nTo get started:")
        print("1. Place your images/videos in the 'data/' directory")
        print("2. Run: python app.py --input data/your_file.jpg --output outputs/result.jpg")
        print("3. Or try: python app.py --webcam for real-time detection")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Run: python setup.py to install dependencies")
        print("2. Run: python test_system.py to check system status")
        print("3. Make sure you have Python 3.8+ installed")


if __name__ == "__main__":
    main() 