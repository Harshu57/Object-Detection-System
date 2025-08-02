"""
Human Emotion and Mood Detection Module
Uses facial expression analysis to detect human emotions and moods.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import config


class EmotionDetector:
    """
    Emotion detection class using facial expression analysis.
    """
    
    def __init__(self):
        """
        Initialize the emotion detector with pre-trained models.
        """
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None
        
        # Simplified emotion labels - only 3 moods
        self.emotion_labels = ['Happy', 'Sad', 'Angry']
        
        # Simple mood mapping
        self.mood_mapping = {
            'Happy': 'Happy',
            'Sad': 'Sad', 
            'Angry': 'Angry'
        }
        
        # Load pre-trained models
        self._load_models()
    
    def _load_models(self):
        """
        Load pre-trained face detection and emotion classification models.
        """
        try:
            # Load OpenCV cascade classifiers
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            self.smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
            
            if self.face_cascade.empty():
                print("Warning: Could not load face cascade classifier")
            
            print("Emotion detector initialized successfully")
            
        except Exception as e:
            print(f"Error loading emotion detection models: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
        
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def analyze_emotion(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Analyze emotion in a face region using simplified feature extraction.
        
        Args:
            face_roi: Face region of interest
        
        Returns:
            Dictionary with emotion probabilities
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize for analysis
        gray = cv2.resize(gray, (64, 64))
        
        # Extract facial features
        features = self._extract_facial_features(gray, face_roi)
        
        # Use simplified emotion prediction
        emotions = self._predict_emotion_from_features(features)
        
        return emotions
    
    def _extract_facial_features(self, gray: np.ndarray, face_roi: np.ndarray) -> Dict:
        """
        Extract facial features for emotion analysis.
        
        Args:
            gray: Grayscale face image
            face_roi: Original face region
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic intensity features
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['min_intensity'] = np.min(gray)
        features['max_intensity'] = np.max(gray)
        
        # Texture and contrast features
        features['entropy'] = self._calculate_entropy(gray)
        features['contrast'] = self._calculate_contrast(gray)
        features['brightness_variance'] = np.var(gray)
        
        # Color features (from original image)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        features['value_mean'] = np.mean(hsv[:, :, 2])
        
        # Facial feature detection
        features.update(self._detect_facial_features(gray, face_roi))
        
        return features
    
    def _detect_facial_features(self, gray: np.ndarray, face_roi: np.ndarray) -> Dict:
        """
        Detect specific facial features like eyes and smile.
        
        Args:
            gray: Grayscale face image
            face_roi: Original face region
        
        Returns:
            Dictionary of facial feature counts
        """
        features = {}
        
        # Detect eyes
        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)
            features['eye_count'] = len(eyes)
        else:
            features['eye_count'] = 0
        
        # Detect smile
        if self.smile_cascade is not None:
            smiles = self.smile_cascade.detectMultiScale(gray, 1.7, 20)
            features['smile_detected'] = len(smiles) > 0
        else:
            features['smile_detected'] = False
        
        # Calculate facial symmetry (simplified)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        if left_half.shape == right_half.shape:
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            features['symmetry_score'] = 1.0 - (symmetry_diff / 255.0)
        else:
            features['symmetry_score'] = 0.5
        
        return features
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate image entropy as a texture measure.
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calculate image contrast.
        """
        return np.std(image)
    
    def _predict_emotion_from_features(self, features: Dict) -> Dict[str, float]:
        """
        Predict emotions based on extracted features with simplified logic.
        
        Args:
            features: Extracted image features
        
        Returns:
            Dictionary with emotion probabilities
        """
        emotions = {}
        
        # Happy: high brightness, smile detected, high symmetry, low contrast
        if (features['mean_intensity'] > 110 and 
            features.get('smile_detected', False) and
            features.get('symmetry_score', 0) > 0.6 and
            features['contrast'] < 35):
            emotions['Happy'] = 0.90
            emotions['Sad'] = 0.05
            emotions['Angry'] = 0.05
        
        # Sad: low brightness, low symmetry, low saturation
        elif (features['mean_intensity'] < 85 and 
              features.get('symmetry_score', 0) < 0.5 and
              features['saturation_mean'] < 70):
            emotions['Sad'] = 0.85
            emotions['Angry'] = 0.10
            emotions['Happy'] = 0.05
        
        # Angry: high contrast, high entropy, low symmetry
        elif (features['contrast'] > 45 and 
              features['entropy'] > 6.5 and
              features.get('symmetry_score', 0) < 0.4):
            emotions['Angry'] = 0.85
            emotions['Sad'] = 0.10
            emotions['Happy'] = 0.05
        
        # Default to Neutral-like (Sad) for balanced features
        else:
            emotions['Sad'] = 0.70
            emotions['Happy'] = 0.15
            emotions['Angry'] = 0.15
        
        return emotions
    
    def detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect emotions for all faces in the image.
        
        Args:
            image: Input image
        
        Returns:
            List of emotion detection results
        """
        results = []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Analyze emotion
            emotions = self.analyze_emotion(face_roi)
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Mood is the same as emotion for simplified system
            mood = dominant_emotion[0]
            
            results.append({
                'bbox': [x, y, x+w, y+h],
                'emotions': emotions,
                'dominant_emotion': dominant_emotion[0],
                'emotion_confidence': dominant_emotion[1],
                'mood': mood,
                'face_roi': face_roi
            })
        
        return results
    
    def draw_emotion_results(self, image: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Draw emotion detection results on the image.
        
        Args:
            image: Input image
            results: Emotion detection results
        
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            emotion = result['dominant_emotion']
            confidence = result['emotion_confidence']
            mood = result['mood']
            
            # Draw face bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Draw mood label with color coding
            mood_color = self._get_mood_color(mood)
            mood_label = f"Mood: {mood}"
            cv2.putText(
                annotated_image,
                mood_label,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                mood_color,
                2
            )
        
        return annotated_image
    
    def _get_mood_color(self, mood: str) -> Tuple[int, int, int]:
        """
        Get color for mood display.
        
        Args:
            mood: Mood classification
        
        Returns:
            BGR color tuple
        """
        mood_colors = {
            'Happy': (0, 255, 0),    # Green
            'Sad': (255, 0, 0),      # Blue
            'Angry': (0, 0, 255),    # Red
        }
        return mood_colors.get(mood, (128, 128, 128))  # Gray default
    
    def get_emotion_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary of detected emotions.
        
        Args:
            results: Emotion detection results
        
        Returns:
            Summary dictionary
        """
        if not results:
            return {'total_faces': 0, 'emotions': {}, 'moods': {}}
        
        emotion_counts = {}
        mood_counts = {}
        
        for result in results:
            emotion = result['dominant_emotion']
            mood = result['mood']
            
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        return {
            'total_faces': len(results),
            'emotions': emotion_counts,
            'moods': mood_counts,
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'None',
            'dominant_mood': max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else 'None'
        }


class AdvancedEmotionDetector(EmotionDetector):
    """
    Advanced emotion detector with more sophisticated analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.emotion_colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
        }
    
    def analyze_emotion_advanced(self, face_roi: np.ndarray) -> Dict[str, float]:
        """
        Advanced emotion analysis using multiple features.
        
        Args:
            face_roi: Face region of interest
        
        Returns:
            Dictionary with emotion probabilities
        """
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Extract multiple features
        features = self._extract_emotion_features(gray, hsv)
        
        # Use feature-based emotion prediction
        emotions = self._predict_emotions_from_features(features)
        
        return emotions
    
    def _extract_emotion_features(self, gray: np.ndarray, hsv: np.ndarray) -> Dict:
        """
        Extract features for emotion analysis.
        
        Args:
            gray: Grayscale image
            hsv: HSV color space image
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic intensity features
        features['mean_intensity'] = np.mean(gray)
        features['std_intensity'] = np.std(gray)
        features['min_intensity'] = np.min(gray)
        features['max_intensity'] = np.max(gray)
        
        # Texture features
        features['entropy'] = self._calculate_entropy(gray)
        features['contrast'] = self._calculate_contrast(gray)
        
        # Color features
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        features['value_mean'] = np.mean(hsv[:, :, 2])
        
        return features
    
    def _predict_emotions_from_features(self, features: Dict) -> Dict[str, float]:
        """
        Predict emotions based on extracted features.
        
        Args:
            features: Extracted image features
        
        Returns:
            Dictionary with emotion probabilities
        """
        emotions = {}
        
        # Happy: high brightness, high saturation, low contrast
        if (features['mean_intensity'] > 120 and 
            features['saturation_mean'] > 80 and 
            features['contrast'] < 30):
            emotions['Happy'] = 0.85
            emotions['Sad'] = 0.10
            emotions['Angry'] = 0.05
        
        # Sad: low brightness, low saturation, low contrast
        elif (features['mean_intensity'] < 80 and 
              features['saturation_mean'] < 60 and 
              features['contrast'] < 25):
            emotions['Sad'] = 0.80
            emotions['Angry'] = 0.15
            emotions['Happy'] = 0.05
        
        # Angry: high contrast, high entropy
        elif (features['contrast'] > 40 and 
              features['entropy'] > 6.0):
            emotions['Angry'] = 0.80
            emotions['Sad'] = 0.15
            emotions['Happy'] = 0.05
        
        # Default to Sad for balanced features
        else:
            emotions['Sad'] = 0.70
            emotions['Happy'] = 0.20
            emotions['Angry'] = 0.10
        
        return emotions


def create_emotion_detector(advanced: bool = True) -> Union[EmotionDetector, AdvancedEmotionDetector]:
    """
    Factory function to create emotion detector.
    
    Args:
        advanced: Whether to use advanced emotion detection
    
    Returns:
        Emotion detector instance
    """
    if advanced:
        return AdvancedEmotionDetector()
    else:
        return EmotionDetector() 