#!/usr/bin/env python3
"""
YOLOv11 Object Detection System - Main Application
A modular object detection system using YOLOv11 for images, videos, and webcam.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLOv11Detector
from src import config
from src import utils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLOv11 Object Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect objects in a single image
  python app.py --input data/image.jpg --output outputs/detected_image.jpg

  # Process all images in a directory
  python app.py --input data/ --output outputs/ --batch

  # Process video file
  python app.py --input data/video.mp4 --output outputs/detected_video.mp4

  # Use webcam for real-time detection
  python app.py --webcam

  # Use webcam with full screen display
  python app.py --webcam --fullscreen

  # Use custom model and confidence threshold
  python app.py --input data/image.jpg --model models/yolov11s.pt --conf 0.5

  # Enable Test Time Augmentation for improved accuracy
  python app.py --webcam --tta

  # Use ensemble models for improved accuracy
  python app.py --input data/image.jpg --ensemble-models models/yolov11s.pt models/yolov11m.pt
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input image/video file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file or directory"
    )
    parser.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Use webcam for real-time detection"
    )
    
    # Processing options
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all supported files in input directory"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Show results in window (for images and videos)"
    )
    
    # Model and detection settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=str(config.MODEL_PATH),
        help=f"Path to YOLOv11 model (default: {config.MODEL_PATH})"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=config.NMS_THRESHOLD,
        help=f"NMS threshold (default: {config.NMS_THRESHOLD})"
    )
    
    # Accuracy improvement features
    parser.add_argument(
        "--ensemble-models",
        nargs='+',
        help="Additional models for ensemble detection"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Test Time Augmentation for improved accuracy"
    )
    parser.add_argument(
        "--calibration-factor",
        type=float,
        default=1.0,
        help="Confidence calibration factor (default: 1.0)"
    )
    parser.add_argument(
        "--no-improved-accuracy",
        action="store_true",
        help="Disable improved accuracy features"
    )
    
    # Webcam settings
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=config.WEBCAM_INDEX,
        help=f"Webcam device index (default: {config.WEBCAM_INDEX})"
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display webcam detection in full screen mode"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=config.TARGET_FPS,
        help=f"Target FPS for webcam detection (default: {config.TARGET_FPS})"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=config.FRAME_SKIP,
        help=f"Number of frames to skip between detections (default: {config.FRAME_SKIP})"
    )
    
    return parser.parse_args()


def validate_input_path(input_path: str) -> bool:
    """Validate input path exists and is supported."""
    path = Path(input_path)
    
    if not path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return False
    
    if path.is_file():
        # Check if it's a supported file type
        suffix = path.suffix.lower()
        if suffix in config.SUPPORTED_IMAGE_FORMATS + config.SUPPORTED_VIDEO_FORMATS:
            return True
        else:
            print(f"Error: Unsupported file type: {suffix}")
            print(f"Supported formats: {config.SUPPORTED_IMAGE_FORMATS + config.SUPPORTED_VIDEO_FORMATS}")
            return False
    
    elif path.is_dir():
        return True
    
    else:
        print(f"Error: Invalid input path: {input_path}")
        return False


def determine_input_type(input_path: str) -> str:
    """Determine if input is image, video, or directory."""
    path = Path(input_path)
    
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in config.SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif suffix in config.SUPPORTED_VIDEO_FORMATS:
            return "video"
        else:
            return "unknown"
    elif path.is_dir():
        return "directory"
    else:
        return "unknown"


def main():
    """Main function to run the YOLOv11 object detection system."""
    args = parse_arguments()
    
    # Validate arguments
    if not args.webcam and not args.input:
        print("Error: Must specify either --input or --webcam")
        sys.exit(1)
    
    if args.webcam and args.input:
        print("Error: Cannot use both --webcam and --input")
        sys.exit(1)
    
    if args.input and not validate_input_path(args.input):
        sys.exit(1)
    
    # Initialize detector
    try:
        print("Initializing YOLOv11 detector...")
        detector = YOLOv11Detector(
            model_path=args.model,
            confidence_threshold=args.conf,
            nms_threshold=args.nms,
            ensemble_models=args.ensemble_models,
            use_tta=args.tta,
            calibration_factor=args.calibration_factor
        )
        print("Detector initialized successfully!")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)
    
    # Process based on input type
    if args.webcam:
        print("=" * 60)
        print("YOLOv11 Webcam Detection")
        print("=" * 60)
        print("Starting webcam detection with YOLOv11 optimizations...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'f' to toggle fullscreen")
        print("  - Press 's' to save screenshot")
        print()
        
        output_path = Path(args.output) if args.output else None
        if output_path:
            print(f"Recording video to: {output_path}")
        
        detector.detect_webcam(
            webcam_index=args.webcam_index,
            output_path=output_path,
            fullscreen=args.fullscreen,
            target_fps=args.target_fps,
            frame_skip=args.frame_skip
        )
    
    elif args.input:
        input_type = determine_input_type(args.input)
        
        if input_type == "image":
            # Single image detection
            input_path = Path(args.input)
            output_path = Path(args.output) if args.output else None
            
            print(f"Processing image: {input_path}")
            result = detector.detect_image(
                input_path,
                output_path,
                show_result=args.show,
                use_improved_accuracy=not args.no_improved_accuracy
            )
            
            if "error" not in result:
                utils.print_detection_summary(
                    result["total_detections"],
                    result["processing_time"],
                    input_path,
                    output_path
                )
            else:
                print(f"Error: {result['error']}")
        
        elif input_type == "video":
            # Video detection
            input_path = Path(args.input)
            output_path = Path(args.output) if args.output else None
            
            print(f"Processing video: {input_path}")
            result = detector.detect_video(
                input_path,
                output_path,
                show_result=args.show
            )
            
            if "error" not in result:
                utils.print_detection_summary(
                    result["total_detections"],
                    result["processing_time"],
                    input_path,
                    output_path
                )
                print(f"Total frames processed: {result['total_frames']}")
            else:
                print(f"Error: {result['error']}")
        
        elif input_type == "directory":
            # Batch processing
            input_dir = Path(args.input)
            output_dir = Path(args.output) if args.output else None
            
            if args.batch:
                print(f"Processing all files in directory: {input_dir}")
                result = detector.batch_detect_images(input_dir, output_dir)
                
                if "error" not in result:
                    print("\n" + "="*50)
                    print("BATCH PROCESSING SUMMARY")
                    print("="*50)
                    print(f"Processed files: {result['processed_files']}/{result['total_files']}")
                    print(f"Total detections: {result['total_detections']}")
                    print(f"Total processing time: {result['total_processing_time']:.2f}s")
                    print(f"Average time per image: {result['average_time_per_image']:.2f}s")
                    print("="*50)
                else:
                    print(f"Error: {result['error']}")
            else:
                print("Error: Directory input requires --batch flag")
                sys.exit(1)
        
        else:
            print(f"Error: Unsupported input type: {input_type}")
            sys.exit(1)


if __name__ == "__main__":
    main() 