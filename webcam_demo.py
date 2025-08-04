#!/usr/bin/env python3
"""
YOLOv11 Webcam Detection Demo
A simple script to run real-time object detection using webcam with YOLOv11.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLOv11Detector
from src import config


def parse_arguments():
    """Parse command line arguments for webcam demo."""
    parser = argparse.ArgumentParser(description="YOLOv11 Webcam Detection Demo")
    parser.add_argument(
        "--target-fps", "-fps",
        type=int,
        default=config.TARGET_FPS,
        help=f"Target FPS for detection (default: {config.TARGET_FPS})"
    )
    parser.add_argument(
        "--frame-skip", "-skip",
        type=int,
        default=config.FRAME_SKIP,
        help=f"Number of frames to skip between detections (default: {config.FRAME_SKIP})"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--fullscreen", "-fs",
        action="store_true",
        help="Start in fullscreen mode"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save output video (optional)"
    )
    return parser.parse_args()


def main():
    """Run YOLOv11 webcam detection demo."""
    args = parse_arguments()
    
    print("=" * 60)
    print("YOLOv11 Webcam Detection Demo")
    print("=" * 60)
    print(f"Target FPS: {args.target_fps}")
    print(f"Frame Skip: {args.frame_skip}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Fullscreen: {args.fullscreen}")
    if args.output:
        print(f"Output Video: {args.output}")
    print()
    
    # Initialize detector with YOLOv11 optimizations
    try:
        print("Initializing YOLOv11 detector...")
        detector = YOLOv11Detector(
            model_path=None,  # Will use default with fallback
            confidence_threshold=args.confidence,
            nms_threshold=config.NMS_THRESHOLD,
            use_tta=False,  # Disable TTA for real-time performance
            calibration_factor=1.0
        )
        print("Detector initialized successfully!")
        print()
        
        print("Starting webcam detection...")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'f' to toggle fullscreen")
        print("  - Press 's' to save screenshot")
        print()
        
        # Start webcam detection with optimized settings
        detector.detect_webcam(
            webcam_index=config.WEBCAM_INDEX,
            output_path=args.output,
            fullscreen=args.fullscreen,
            target_fps=args.target_fps,
            frame_skip=args.frame_skip
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your webcam is connected and accessible.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 