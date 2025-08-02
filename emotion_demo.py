#!/usr/bin/env python3
"""
Emotion Detection Demo
Demonstrates human emotion and mood detection capabilities.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.emotion_detector import create_emotion_detector
from src.detector import YOLOv8Detector
import cv2
import numpy as np


def demo_emotion_detection():
    """Demo emotion detection on webcam."""
    print("🎭 Emotion Detection Demo")
    print("=" * 50)
    print("This demo will detect human emotions and moods in real-time.")
    print("Press 'q' to quit, 'e' to toggle emotion detection, 'a' to toggle advanced mode")
    print()
    
    # Initialize detector with emotion detection
    detector = YOLOv8Detector(
        enable_emotion_detection=True,
        advanced_emotion=True,
        confidence_threshold=0.3
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create window
    window_name = "Emotion Detection Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    emotion_enabled = True
    advanced_mode = True
    frame_count = 0
    
    print("Starting emotion detection...")
    print("Look at the camera and show different expressions!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Perform object detection
        results = detector.model(
            frame,
            conf=detector.confidence_threshold,
            iou=detector.nms_threshold,
            max_det=10
        )
        
        # Draw object detections
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = detector.class_names.get(class_id, f"class_{class_id}")
                    
                    # Only draw bounding boxes for humans and objects
                    if class_name == "person" or confidence > 0.5:
                        color = (255, 255, 0)  # Yellow for objects
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(
                            annotated_frame,
                            f"{class_name}: {confidence:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1
                        )
        
        # Perform emotion detection if enabled
        if emotion_enabled and detector.emotion_detector is not None:
            try:
                emotion_results = detector.emotion_detector.detect_emotions(frame)
                if emotion_results:
                    annotated_frame = detector.emotion_detector.draw_emotion_results(annotated_frame, emotion_results)
                    
                    # Print emotion summary every 30 frames
                    if frame_count % 30 == 0:
                        summary = detector.emotion_detector.get_emotion_summary(emotion_results)
                        if summary['total_faces'] > 0:
                            print(f"😊 Detected: {summary['dominant_emotion']} | Mood: {summary['dominant_mood']} | Faces: {summary['total_faces']}")
            except Exception as e:
                pass
        
        # Add status text
        status_text = f"Emotion: {'ON' if emotion_enabled else 'OFF'} | Mode: {'Advanced' if advanced_mode else 'Basic'}"
        cv2.putText(
            annotated_frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # Add instructions
        cv2.putText(
            annotated_frame,
            "Press 'q' to quit, 'e' to toggle emotion, 'a' to toggle mode",
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display frame
        cv2.imshow(window_name, annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            emotion_enabled = not emotion_enabled
            print(f"Emotion detection: {'ENABLED' if emotion_enabled else 'DISABLED'}")
        elif key == ord('a'):
            advanced_mode = not advanced_mode
            # Reinitialize emotion detector with new mode
            try:
                detector.emotion_detector = create_emotion_detector(advanced=advanced_mode)
                print(f"Emotion mode: {'ADVANCED' if advanced_mode else 'BASIC'}")
            except Exception as e:
                print(f"Failed to switch emotion mode: {e}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Emotion detection demo ended.")


def demo_emotion_analysis():
    """Demo emotion analysis on sample images."""
    print("📸 Emotion Analysis Demo")
    print("=" * 50)
    
    # Initialize emotion detector
    emotion_detector = create_emotion_detector(advanced=True)
    
    # Sample images (you can add your own images to test)
    sample_images = [
        "data/sample_face.jpg",
        "data/happy_face.jpg",
        "data/sad_face.jpg"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            print(f"\nAnalyzing: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Detect emotions
            emotion_results = emotion_detector.detect_emotions(image)
            
            if emotion_results:
                # Get summary
                summary = emotion_detector.get_emotion_summary(emotion_results)
                
                print(f"  Faces detected: {summary['total_faces']}")
                print(f"  Dominant emotion: {summary['dominant_emotion']}")
                print(f"  Overall mood: {summary['dominant_mood']}")
                
                # Show detailed results
                for i, result in enumerate(emotion_results):
                    print(f"  Face {i+1}: {result['dominant_emotion']} ({result['emotion_confidence']:.2f}) - {result['mood']}")
                
                # Draw results
                annotated_image = emotion_detector.draw_emotion_results(image, emotion_results)
                
                # Save result
                output_path = f"outputs/emotion_{Path(image_path).stem}.jpg"
                cv2.imwrite(output_path, annotated_image)
                print(f"  Result saved: {output_path}")
            else:
                print("  No faces detected")
        else:
            print(f"Sample image not found: {image_path}")
    
    print("\nEmotion analysis demo completed.")


def main():
    """Main function for emotion detection demo."""
    print("🎭 YOLOv8 Emotion Detection System")
    print("=" * 60)
    print("This system can detect human emotions and moods in addition to objects.")
    print()
    
    print("Demo Options:")
    print("1. Real-time emotion detection (webcam)")
    print("2. Emotion analysis on sample images")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        demo_emotion_detection()
    elif choice == "2":
        demo_emotion_analysis()
    elif choice == "3":
        demo_emotion_analysis()
        print("\n" + "="*60)
        demo_emotion_detection()
    else:
        print("Invalid choice. Running real-time demo...")
        demo_emotion_detection()


if __name__ == "__main__":
    main() 