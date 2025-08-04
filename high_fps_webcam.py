#!/usr/bin/env python3
"""
High-FPS YOLOv11 Webcam Detection
Optimized for maximum frame rate with minimal latency.
"""

import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from threading import Thread, Lock
from queue import Queue

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import YOLOv11Detector
from src import config


class HighFPSWebcamDetector:
    """High-FPS webcam detector with threading optimizations."""
    
    def __init__(self, model_path=None, confidence_threshold=0.25):
        """Initialize the high-FPS detector."""
        self.detector = YOLOv11Detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            use_tta=False,
            calibration_factor=1.0
        )
        
        # Threading variables
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = False
        self.lock = Lock()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def capture_frames(self, webcam_index=0):
        """Capture frames in a separate thread."""
        cap = cv2.VideoCapture(webcam_index)
        
        # Optimize camera settings for high FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Clear old frames and add new one
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(frame)
            
        cap.release()
    
    def process_detections(self):
        """Process detections in a separate thread."""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Perform detection
                results = self.detector.model(
                    frame,
                    conf=self.detector.confidence_threshold,
                    iou=self.detector.nms_threshold,
                    max_det=config.MAX_DETECTIONS,
                    verbose=False,
                    stream=True
                )
                
                # Process results
                annotated_frame = frame.copy()
                detections_count = 0
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Get class and confidence
                            class_id = int(box.cls[0].cpu().numpy())
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.detector.class_names.get(class_id, f"class_{class_id}")
                            
                            # Get color for this class
                            color = (0, 255, 0)  # Green for high visibility
                            
                            # Draw bounding box (simplified for speed)
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}", 
                                      (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            detections_count += 1
                
                # Add FPS overlay
                cv2.putText(annotated_frame, f"FPS: {self.fps:.1f} | Detections: {detections_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Clear old results and add new one
                if not self.result_queue.empty():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(annotated_frame)
    
    def run(self, webcam_index=0, target_fps=60):
        """Run high-FPS webcam detection."""
        self.running = True
        
        # Start capture thread
        capture_thread = Thread(target=self.capture_frames, args=(webcam_index,))
        capture_thread.daemon = True
        capture_thread.start()
        
        # Start detection thread
        detection_thread = Thread(target=self.process_detections)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Create window
        window_name = "High-FPS YOLOv11 Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("High-FPS webcam detection started!")
        print("Controls: 'q' to quit, 'f' for fullscreen")
        
        frame_count = 0
        last_time = time.time()
        
        while True:
            if not self.result_queue.empty():
                frame = self.result_queue.get()
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if frame_count % 30 == 0:
                    self.fps = 30 / (current_time - last_time)
                    last_time = current_time
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    # Toggle fullscreen
                    current_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    if current_prop == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Cleanup
        self.running = False
        cv2.destroyAllWindows()
        print(f"High-FPS detection stopped. Average FPS: {self.fps:.1f}")


def main():
    """Run high-FPS webcam detection."""
    parser = argparse.ArgumentParser(description="High-FPS YOLOv11 Webcam Detection")
    parser.add_argument(
        "--target-fps", "-fps",
        type=int,
        default=60,
        help="Target FPS for detection (default: 60)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--webcam-index", "-i",
        type=int,
        default=0,
        help="Webcam device index (default: 0)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("High-FPS YOLOv11 Webcam Detection")
    print("=" * 60)
    print(f"Target FPS: {args.target_fps}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Webcam Index: {args.webcam_index}")
    print()
    
    try:
        detector = HighFPSWebcamDetector(
            confidence_threshold=args.confidence
        )
        detector.run(
            webcam_index=args.webcam_index,
            target_fps=args.target_fps
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 