#!/usr/bin/env python3
"""
Example script demonstrating YOLOv8 Object Detection System usage
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLOv8Detector
from src import config
from src import utils


def example_single_image():
    """Example: Detect objects in a single image."""
    print("=== Single Image Detection Example ===")
    
    # Initialize detector
    detector = YOLOv8Detector(
        confidence_threshold=0.3,
        nms_threshold=0.45
    )
    
    # Example image path (you would replace this with your actual image)
    image_path = "data/sample_image.jpg"
    output_path = "outputs/detected_sample.jpg"
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please place a sample image in the data/ directory")
        return
    
    # Perform detection
    result = detector.detect_image(image_path, output_path, show_result=False)
    
    if "error" not in result:
        print(f"Detection completed successfully!")
        print(f"Total detections: {result['total_detections']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        # Print detection details
        for i, detection in enumerate(result['detections']):
            print(f"  Detection {i+1}: {detection['class_name']} "
                  f"(confidence: {detection['confidence']:.2f})")
    else:
        print(f"Error: {result['error']}")


def example_batch_processing():
    """Example: Process multiple images in a directory."""
    print("\n=== Batch Processing Example ===")
    
    # Initialize detector
    detector = YOLOv8Detector(confidence_threshold=0.25)
    
    # Process all images in data directory
    input_dir = "data"
    output_dir = "outputs/batch_results"
    
    if not Path(input_dir).exists():
        print(f"Input directory not found: {input_dir}")
        print("Please create a data/ directory with some images")
        return
    
    # Perform batch detection
    result = detector.batch_detect_images(input_dir, output_dir)
    
    if "error" not in result:
        print(f"Batch processing completed!")
        print(f"Processed files: {result['processed_files']}/{result['total_files']}")
        print(f"Total detections: {result['total_detections']}")
        print(f"Total time: {result['total_processing_time']:.2f} seconds")
        print(f"Average time per image: {result['average_time_per_image']:.2f} seconds")
    else:
        print(f"Error: {result['error']}")


def example_video_processing():
    """Example: Process a video file."""
    print("\n=== Video Processing Example ===")
    
    # Initialize detector
    detector = YOLOv8Detector(confidence_threshold=0.3)
    
    # Example video path
    video_path = "data/sample_video.mp4"
    output_path = "outputs/detected_video.mp4"
    
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        print("Please place a sample video in the data/ directory")
        return
    
    # Perform video detection
    result = detector.detect_video(video_path, output_path, show_result=False)
    
    if "error" not in result:
        print(f"Video processing completed!")
        print(f"Total detections: {result['total_detections']}")
        print(f"Total frames: {result['total_frames']}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Video properties: {result['video_properties']}")
    else:
        print(f"Error: {result['error']}")


def example_webcam_detection():
    """Example: Real-time webcam detection."""
    print("\n=== Webcam Detection Example ===")
    print("This will start webcam detection. Press 'q' to quit.")
    
    # Initialize detector
    detector = YOLOv8Detector(confidence_threshold=0.3)
    
    # Start webcam detection
    try:
        detector.detect_webcam(output_path="outputs/webcam_recording.mp4")
    except KeyboardInterrupt:
        print("\nWebcam detection stopped by user.")
    except Exception as e:
        print(f"Error during webcam detection: {e}")


def example_custom_model():
    """Example: Using a custom model."""
    print("\n=== Custom Model Example ===")
    
    # Initialize detector with custom model
    # You can specify different YOLOv8 models: yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.
    detector = YOLOv8Detector(
        model_path="yolov8s.pt",  # Using YOLOv8 small model
        confidence_threshold=0.4,
        nms_threshold=0.5
    )
    
    print("Custom model loaded successfully!")
    print(f"Available classes: {len(detector.class_names)}")
    
    # Example usage with custom model
    image_path = "data/sample_image.jpg"
    if Path(image_path).exists():
        result = detector.detect_image(image_path, "outputs/custom_model_result.jpg")
        if "error" not in result:
            print(f"Custom model detection completed!")
            print(f"Detections: {result['total_detections']}")
        else:
            print(f"Error: {result['error']}")


def main():
    """Run all examples."""
    print("YOLOv8 Object Detection System - Examples")
    print("=" * 50)
    
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/batch_results").mkdir(exist_ok=True)
    
    # Run examples
    example_single_image()
    example_batch_processing()
    example_video_processing()
    example_custom_model()
    
    # Webcam example (commented out by default as it requires user interaction)
    # Uncomment the line below to enable webcam detection
    # example_webcam_detection()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the outputs/ directory for results.")
    print("For webcam detection, uncomment the webcam example in the code.")


if __name__ == "__main__":
    main() 