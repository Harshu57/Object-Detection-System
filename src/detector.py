"""
YOLOv8 Object Detector Implementation
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
from ultralytics import YOLO
import config
import utils
from .emotion_detector import create_emotion_detector

class YOLOv8Detector:
    """
    YOLOv8 Object Detector class for processing images and videos.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.NMS_THRESHOLD,
        ensemble_models: Optional[List[str]] = None,
        use_tta: bool = False,
        calibration_factor: float = 1.0,
        enable_emotion_detection: bool = True,
        advanced_emotion: bool = True
    ):
        """
        Initialize the YOLOv8 detector with improved accuracy features.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            ensemble_models: List of additional models for ensemble detection
            use_tta: Whether to use Test Time Augmentation
            calibration_factor: Confidence calibration factor
            enable_emotion_detection: Whether to enable emotion detection
            advanced_emotion: Whether to use advanced emotion detection
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_tta = use_tta
        self.calibration_factor = calibration_factor
        self.enable_emotion_detection = enable_emotion_detection
        
        # Set default model path if not provided
        if model_path is None:
            model_path = config.MODEL_PATH
        
        # Load primary model
        self.model = self._load_model(model_path)
        
        # Load ensemble models if provided
        self.ensemble_models = []
        if ensemble_models:
            for model_path in ensemble_models:
                try:
                    ensemble_model = self._load_model(model_path)
                    self.ensemble_models.append(ensemble_model)
                    print(f"Loaded ensemble model: {model_path}")
                except Exception as e:
                    print(f"Failed to load ensemble model {model_path}: {e}")
        
        # Initialize emotion detector if enabled
        self.emotion_detector = None
        if self.enable_emotion_detection:
            try:
                self.emotion_detector = create_emotion_detector(advanced=advanced_emotion)
                print("Emotion detector initialized successfully")
            except Exception as e:
                print(f"Failed to initialize emotion detector: {e}")
                self.emotion_detector = None
        
        # Store class names
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        print(f"YOLOv8 model loaded successfully from {model_path}")
        print(f"Available classes: {len(self.class_names)}")
        print(f"Ensemble models: {len(self.ensemble_models)}")
        print(f"TTA enabled: {self.use_tta}")
        print(f"Calibration factor: {self.calibration_factor}")
        print(f"Emotion detection enabled: {self.emotion_detector is not None}")
    
    def _load_model(self, model_path: Union[str, Path]) -> YOLO:
        """
        Load YOLOv8 model from path.
        
        Args:
            model_path: Path to model weights
        
        Returns:
            Loaded YOLO model
        """
        try:
            model = YOLO(str(model_path))
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Attempting to download default YOLOv8n model...")
            try:
                model = YOLO("yolov8n.pt")
                return model
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
    
    def _apply_confidence_calibration(self, confidence: float) -> float:
        """
        Apply confidence calibration to improve accuracy.
        
        Args:
            confidence: Raw confidence score
        
        Returns:
            Calibrated confidence score
        """
        # Apply calibration factor
        calibrated = confidence * self.calibration_factor
        
        # Apply sigmoid-like calibration for better probability estimates
        calibrated = 1 / (1 + np.exp(-5 * (calibrated - 0.5)))
        
        return min(max(calibrated, 0.0), 1.0)
    
    def _ensemble_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Perform ensemble detection using multiple models.
        
        Args:
            image: Input image
        
        Returns:
            List of ensemble detection results
        """
        all_detections = []
        
        # Get primary model detections
        primary_results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            max_det=config.MAX_DETECTIONS
        )
        
        for result in primary_results:
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy(),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'model_weight': 1.0
                    }
                    all_detections.append(detection)
        
        # Get ensemble model detections
        for i, ensemble_model in enumerate(self.ensemble_models):
            try:
                ensemble_results = ensemble_model(
                    image,
                    conf=self.confidence_threshold * 0.8,  # Lower threshold for ensemble
                    iou=self.nms_threshold,
                    max_det=config.MAX_DETECTIONS
                )
                
                for result in ensemble_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            detection = {
                                'bbox': box.xyxy[0].cpu().numpy(),
                                'confidence': float(box.conf[0].cpu().numpy()),
                                'class_id': int(box.cls[0].cpu().numpy()),
                                'model_weight': 0.5  # Lower weight for ensemble models
                            }
                            all_detections.append(detection)
            except Exception as e:
                print(f"Ensemble model {i} failed: {e}")
        
        return all_detections
    
    def _merge_ensemble_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge ensemble detections using weighted averaging.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Merged detection results
        """
        if not detections:
            return []
        
        # Group detections by class and spatial proximity
        merged_detections = []
        used_indices = set()
        
        for i, det1 in enumerate(detections):
            if i in used_indices:
                continue
            
            similar_detections = [det1]
            used_indices.add(i)
            
            # Find similar detections
            for j, det2 in enumerate(detections):
                if j in used_indices:
                    continue
                
                # Check if same class and spatially close
                if (det1['class_id'] == det2['class_id'] and 
                    self._calculate_iou(det1['bbox'], det2['bbox']) > 0.3):
                    similar_detections.append(det2)
                    used_indices.add(j)
            
            # Merge similar detections
            if len(similar_detections) > 1:
                merged = self._weighted_average_detections(similar_detections)
                merged_detections.append(merged)
            else:
                merged_detections.append(det1)
        
        return merged_detections
    
    def _calculate_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
        
        Returns:
            IoU value
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _weighted_average_detections(self, detections: List[Dict]) -> Dict:
        """
        Calculate weighted average of multiple detections.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Weighted average detection
        """
        total_weight = sum(det['model_weight'] for det in detections)
        
        # Weighted average of bounding boxes
        weighted_bbox = np.zeros(4)
        weighted_confidence = 0
        
        for det in detections:
            weight = det['model_weight'] / total_weight
            weighted_bbox += det['bbox'] * weight
            weighted_confidence += det['confidence'] * weight
        
        return {
            'bbox': weighted_bbox,
            'confidence': weighted_confidence,
            'class_id': detections[0]['class_id'],  # Use class from first detection
            'model_weight': total_weight
        }
    
    def _apply_test_time_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply Test Time Augmentation for improved accuracy.
        
        Args:
            image: Input image
        
        Returns:
            List of augmented images
        """
        augmented_images = [image]
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        augmented_images.append(bright)
        
        # Contrast adjustment
        contrast = cv2.convertScaleAbs(image, alpha=0.8, beta=0)
        augmented_images.append(contrast)
        
        return augmented_images
    
    def detect_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_result: bool = False,
        use_improved_accuracy: bool = True
    ) -> Dict[str, Any]:
        """
        Perform object detection on a single image with improved accuracy.
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_result: Whether to display the result
            use_improved_accuracy: Whether to use accuracy improvement features
        
        Returns:
            Dictionary containing detection results and metadata
        """
        start_time = time.time()
        
        # Load image
        image = utils.load_image(image_path)
        if image is None:
            return {"error": "Failed to load image"}
        
        detections = []
        
        if use_improved_accuracy and (self.ensemble_models or self.use_tta):
            # Use improved accuracy features
            if self.ensemble_models:
                # Ensemble detection
                all_detections = self._ensemble_detection(image)
                merged_detections = self._merge_ensemble_detections(all_detections)
                
                for detection in merged_detections:
                    # Apply confidence calibration
                    calibrated_confidence = self._apply_confidence_calibration(detection['confidence'])
                    
                    if calibrated_confidence >= self.confidence_threshold:
                        detections.append({
                            "class_id": detection['class_id'],
                            "class_name": self.class_names.get(detection['class_id'], f"class_{detection['class_id']}"),
                            "confidence": calibrated_confidence,
                            "bbox": detection['bbox'].tolist(),
                            "raw_confidence": detection['confidence']
                        })
            
            elif self.use_tta:
                # Test Time Augmentation
                augmented_images = self._apply_test_time_augmentation(image)
                all_detections = []
                
                for aug_image in augmented_images:
                    results = self.model(
                        aug_image,
                        conf=self.confidence_threshold * 0.7,  # Lower threshold for TTA
                        iou=self.nms_threshold,
                        max_det=config.MAX_DETECTIONS
                    )
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                detection = {
                                    'bbox': box.xyxy[0].cpu().numpy(),
                                    'confidence': float(box.conf[0].cpu().numpy()),
                                    'class_id': int(box.cls[0].cpu().numpy())
                                }
                                all_detections.append(detection)
                
                # Merge TTA detections
                if all_detections:
                    merged_detections = self._merge_ensemble_detections(all_detections)
                    
                    for detection in merged_detections:
                        calibrated_confidence = self._apply_confidence_calibration(detection['confidence'])
                        
                        if calibrated_confidence >= self.confidence_threshold:
                            detections.append({
                                "class_id": detection['class_id'],
                                "class_name": self.class_names.get(detection['class_id'], f"class_{detection['class_id']}"),
                                "confidence": calibrated_confidence,
                                "bbox": detection['bbox'].tolist(),
                                "raw_confidence": detection['confidence']
                            })
            
            else:
                # Standard detection with confidence calibration
                results = self.model(
                    image,
                    conf=self.confidence_threshold,
                    iou=self.nms_threshold,
                    max_det=config.MAX_DETECTIONS
                )
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            raw_confidence = float(box.conf[0].cpu().numpy())
                            calibrated_confidence = self._apply_confidence_calibration(raw_confidence)
                            
                            if calibrated_confidence >= self.confidence_threshold:
                                class_id = int(box.cls[0].cpu().numpy())
                                detections.append({
                                    "class_id": class_id,
                                    "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                                    "confidence": calibrated_confidence,
                                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                                    "raw_confidence": raw_confidence
                                })
        else:
            # Standard detection (original method)
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=config.MAX_DETECTIONS
            )
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detections.append({
                            "class_id": class_id,
                            "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                            "confidence": confidence,
                            "bbox": box.xyxy[0].cpu().numpy().tolist()
                        })
        
        # Process results and draw bounding boxes
        annotated_image = image.copy()
        
        # Perform emotion detection if enabled
        emotion_results = []
        if self.emotion_detector is not None:
            try:
                emotion_results = self.emotion_detector.detect_emotions(image)
                if emotion_results:
                    annotated_image = self.emotion_detector.draw_emotion_results(annotated_image, emotion_results)
            except Exception as e:
                print(f"Emotion detection failed: {e}")
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            class_id = detection["class_id"]
            
            # Get color for this class
            color = utils.get_color_for_class(class_id)
            
            # Draw bounding box
            annotated_image = utils.draw_bounding_box(
                annotated_image,
                bbox,
                class_name,
                confidence,
                color,
                class_id
            )
        
        processing_time = time.time() - start_time
        
        # Save output if path provided
        if output_path is not None:
            utils.save_image(annotated_image, output_path)
        
        # Show result if requested
        if show_result:
            cv2.imshow("YOLOv8 Detection", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            "detections": detections,
            "total_detections": len(detections),
            "processing_time": processing_time,
            "input_path": str(image_path),
            "output_path": str(output_path) if output_path else None,
            "image_shape": image.shape,
            "accuracy_features": {
                "ensemble_models": len(self.ensemble_models),
                "tta_enabled": self.use_tta,
                "calibration_factor": self.calibration_factor,
                "improved_accuracy": use_improved_accuracy
            },
            "emotion_detection": {
                "enabled": self.emotion_detector is not None,
                "results": emotion_results,
                "total_faces": len(emotion_results),
                "summary": self.emotion_detector.get_emotion_summary(emotion_results) if self.emotion_detector else {}
            }
        }
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_result: bool = False
    ) -> Dict[str, Any]:
        """
        Perform object detection on a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_result: Whether to display the result
        
        Returns:
            Dictionary containing detection results and metadata
        """
        start_time = time.time()
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Failed to open video file"}
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer if output path provided
        video_writer = None
        if output_path is not None:
            video_writer = utils.create_video_writer(
                output_path, frame_width, frame_height, fps
            )
            if video_writer is None:
                cap.release()
                return {"error": "Failed to create video writer"}
        
        frame_count = 0
        total_detections = 0
        
        print(f"Processing video: {video_path}")
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection on frame
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=config.MAX_DETECTIONS
            )
            
            # Process results
            annotated_frame = frame.copy()
            frame_detections = 0
            
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
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # Get color for this class
                        color = utils.get_color_for_class(class_id)
                        
                        # Draw bounding box
                        annotated_frame = utils.draw_bounding_box(
                            annotated_frame,
                            [x1, y1, x2, y2],
                            class_name,
                            confidence,
                            color,
                            class_id
                        )
                        
                        frame_detections += 1
            
            total_detections += frame_detections
            
            # Write frame to output video
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Show frame if requested
            if show_result:
                cv2.imshow("YOLOv8 Video Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 30 == 0:  # Print every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if show_result:
            cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        return {
            "total_detections": total_detections,
            "total_frames": frame_count,
            "processing_time": processing_time,
            "input_path": str(video_path),
            "output_path": str(output_path) if output_path else None,
            "video_properties": {
                "width": frame_width,
                "height": frame_height,
                "fps": fps,
                "total_frames": total_frames
            }
        }
    
    def detect_webcam(
        self,
        webcam_index: int = config.WEBCAM_INDEX,
        output_path: Optional[Union[str, Path]] = None,
        fullscreen: bool = False
    ) -> None:
        """
        Perform real-time object detection using webcam.
        
        Args:
            webcam_index: Webcam device index
            output_path: Path to save output video (optional)
            fullscreen: Whether to display in full screen mode
        """
        # Open webcam
        cap = cv2.VideoCapture(webcam_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {webcam_index}")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        
        # Get actual frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if output path provided
        video_writer = None
        if output_path is not None:
            video_writer = utils.create_video_writer(
                output_path, frame_width, frame_height
            )
        
        # Create window and set properties
        window_name = "YOLOv8 Webcam Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        if fullscreen:
            # Set window to full screen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Full screen mode enabled. Press 'q' to quit or 'f' to toggle fullscreen.")
        else:
            print("Webcam detection started. Press 'q' to quit or 'f' to toggle fullscreen.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Perform detection on frame
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                max_det=config.MAX_DETECTIONS
            )
            
            # Process results
            annotated_frame = frame.copy()
            
            # Perform emotion detection if enabled
            if self.emotion_detector is not None:
                try:
                    emotion_results = self.emotion_detector.detect_emotions(frame)
                    if emotion_results:
                        annotated_frame = self.emotion_detector.draw_emotion_results(annotated_frame, emotion_results)
                        
                        # Print emotion summary every 30 frames
                        if frame_count % 30 == 0:
                            summary = self.emotion_detector.get_emotion_summary(emotion_results)
                            if summary['total_faces'] > 0:
                                print(f"Emotions detected: {summary['dominant_emotion']} (Mood: {summary['dominant_mood']})")
                except Exception as e:
                    pass  # Don't stop webcam for emotion detection errors
            
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
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # Get color for this class
                        color = utils.get_color_for_class(class_id)
                        
                        # Draw bounding box
                        annotated_frame = utils.draw_bounding_box(
                            annotated_frame,
                            [x1, y1, x2, y2],
                            class_name,
                            confidence,
                            color,
                            class_id
                        )
            
            # Write frame to output video
            if video_writer is not None:
                video_writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow(window_name, annotated_frame)
            
            # Check for quit command or fullscreen toggle
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Full screen mode enabled")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("Full screen mode disabled")
        
        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")
    
    def batch_detect_images(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Perform object detection on all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images (optional)
        
        Returns:
            Dictionary containing batch processing results
        """
        input_dir = Path(input_dir)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Get all supported image files
        image_files = utils.get_supported_files(
            input_dir, config.SUPPORTED_IMAGE_FORMATS
        )
        
        if not image_files:
            return {"error": f"No supported image files found in {input_dir}"}
        
        print(f"Found {len(image_files)} images to process")
        
        total_detections = 0
        total_processing_time = 0
        processed_files = 0
        
        for image_file in image_files:
            print(f"Processing: {image_file.name}")
            
            # Determine output path
            if output_dir is not None:
                output_path = output_dir / f"detected_{image_file.name}"
            else:
                output_path = None
            
            # Perform detection
            result = self.detect_image(image_file, output_path)
            
            if "error" not in result:
                total_detections += result["total_detections"]
                total_processing_time += result["processing_time"]
                processed_files += 1
                
                print(f"  - Detections: {result['total_detections']}")
                print(f"  - Time: {result['processing_time']:.2f}s")
            else:
                print(f"  - Error: {result['error']}")
        
        return {
            "processed_files": processed_files,
            "total_files": len(image_files),
            "total_detections": total_detections,
            "total_processing_time": total_processing_time,
            "average_time_per_image": total_processing_time / processed_files if processed_files > 0 else 0
        } 