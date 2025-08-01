"""
Utility functions for YOLOv8 Object Detection System
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import config

def draw_bounding_box(
    image: np.ndarray,
    box: List[float],
    label: str,
    confidence: float,
    color: Tuple[int, int, int],
    class_id: int
) -> np.ndarray:
    """
    Draw a bounding box with label and confidence score on the image.
    
    Args:
        image: Input image as numpy array
        box: Bounding box coordinates [x1, y1, x2, y2]
        label: Class label
        confidence: Detection confidence score
        color: BGR color tuple for the box
        class_id: Class ID for color selection
    
    Returns:
        Image with drawn bounding box
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Draw bounding box
    cv2.rectangle(
        image, 
        (x1, y1), 
        (x2, y2), 
        color, 
        config.BOX_THICKNESS
    )
    
    # Prepare label text
    label_text = f"{label}: {confidence:.2f}"
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        config.TEXT_SCALE, 
        config.TEXT_THICKNESS
    )
    
    # Draw label background
    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 10),
        (x1 + text_width, y1),
        color,
        -1  # Filled rectangle
    )
    
    # Draw label text
    cv2.putText(
        image,
        label_text,
        (x1, y1 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.TEXT_SCALE,
        config.TEXT_COLOR,
        config.TEXT_THICKNESS
    )
    
    return image

def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """
    Get a color for a specific class ID.
    
    Args:
        class_id: Class ID
    
    Returns:
        BGR color tuple
    """
    return config.BOX_COLORS[class_id % len(config.BOX_COLORS)]

def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Loaded image as numpy array or None if failed
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save an image to file.
    
    Args:
        image: Image to save as numpy array
        output_path: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def get_supported_files(directory: Union[str, Path], file_types: List[str]) -> List[Path]:
    """
    Get all supported files from a directory.
    
    Args:
        directory: Directory to search
        file_types: List of supported file extensions
    
    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    files = []
    for file_type in file_types:
        files.extend(directory.glob(f"*{file_type}"))
        files.extend(directory.glob(f"*{file_type.upper()}"))
    
    return sorted(files)

def create_video_writer(
    output_path: Union[str, Path],
    frame_width: int,
    frame_height: int,
    fps: int = config.VIDEO_FPS
) -> Optional[cv2.VideoWriter]:
    """
    Create a video writer for output video.
    
    Args:
        output_path: Output video file path
        frame_width: Frame width
        frame_height: Frame height
        fps: Frames per second
    
    Returns:
        VideoWriter object or None if failed
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_FOURCC)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        return writer
    except Exception as e:
        print(f"Error creating video writer: {e}")
        return None

def format_time(seconds: float) -> str:
    """
    Format time in seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def print_detection_summary(
    total_detections: int,
    processing_time: float,
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Print a summary of the detection results.
    
    Args:
        total_detections: Total number of detections
        processing_time: Time taken for processing
        input_path: Input file path
        output_path: Output file path
    """
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total Detections: {total_detections}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print("="*50) 