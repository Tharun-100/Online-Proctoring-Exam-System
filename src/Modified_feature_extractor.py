"""
Feature Extraction Module for Fraud Detection System
Extracts 37 features from input images for fraud detection

This module provides robust feature extraction using:
- MediaPipe (face detection, face mesh, hands, pose)
- YOLO (phone detection)
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
import math
import logging
import os
from pathlib import Path
from contextlib import contextmanager

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LANDMARK CONSTANTS (Organized and Centralized)
# ============================================================================
class MediaPipeLandmarks:
    """MediaPipe landmark indices for face mesh"""
    # Eyes
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Iris (pupils)
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    
    # Face features
    NOSE_TIP = 4
    NOSE_ROOT = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    
    # Mouth region
    MOUTH_LANDMARKS = [13, 14, 308, 324, 318]


class HeadPosePoints:
    """3D model points for head pose estimation (in mm)"""
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -63.6, -12.5),    # Chin
        (-43.3, 32.7, -26.0),   # Left eye
        (43.3, 32.7, -26.0),    # Right eye
        (-28.9, -28.9, -24.1),  # Left mouth
        (28.9, -28.9, -24.1)    # Right mouth
    ], dtype=np.float64)
    
    LANDMARK_INDICES = [
        MediaPipeLandmarks.NOSE_ROOT,
        MediaPipeLandmarks.CHIN,
        MediaPipeLandmarks.LEFT_EYE_CORNER,
        MediaPipeLandmarks.RIGHT_EYE_CORNER,
        MediaPipeLandmarks.LEFT_MOUTH,
        MediaPipeLandmarks.RIGHT_MOUTH
    ]


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
class DetectionConfig:
    """Detection confidence thresholds"""
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_PRESENCE_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Gaze detection thresholds
    GAZE_CENTER_THRESHOLD = 80
    GAZE_DIRECTION_THRESHOLD = 100
    HEAD_POSE_THRESHOLD = 20
    
    # Hand-face interaction
    HAND_FACE_PROXIMITY_THRESHOLD = 1.5
    
    # Gaze on script
    SCRIPT_REGION_Y_THRESHOLD = 0.6


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================
class FeatureExtractor:
    """
    Robust feature extraction for fraud detection in online exams.
    Extracts 37 features from images including:
    - Face detection and landmarks
    - Eye and gaze features
    - Hand detection and interactions
    - Head pose estimation
    - Phone detection
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\src\models",
        use_phone_detection: bool = True,
        config: Optional[DetectionConfig] = None
    ):
        """
        Initialize feature extractor with MediaPipe and YOLO models.
        
        Args:
            model_dir: Path to directory containing model files
            use_phone_detection: Whether to load YOLO for phone detection
            config: DetectionConfig object for thresholds
            
        Raises:
            FileNotFoundError: If model directory or required files not found
            RuntimeError: If model initialization fails
        """
        self.model_dir = self._resolve_model_dir(model_dir)
        self.use_phone_detection = use_phone_detection
        self.config = config or DetectionConfig()
        
        # Validate model directory
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Initialize models with error handling
        try:
            self._initialize_mediapipe_models()
            logger.info("✓ MediaPipe models initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize MediaPipe models: {e}")
            raise
        
        # Optional YOLO initialization
        if self.use_phone_detection:
            try:
                self._initialize_yolo_model()
                logger.info("✓ YOLO phone detection model initialized")
            except Exception as e:
                logger.warning(f"Phone detection disabled: {e}")
                self.phone_model = None
        else:
            self.phone_model = None
    
    @staticmethod
    def _resolve_model_dir(model_dir: Optional[str]) -> Path:
        """
        Resolve model directory with fallback options.
        
        Priority:
        1. Explicit parameter
        2. Environment variable 'FRAUD_DETECTION_MODEL_DIR'
        3. Relative path './models' from current directory
        4. Raise error if none exist
        """
        if model_dir:
            path = Path(model_dir)
        else:
            # Try environment variable
            env_path = os.getenv('FRAUD_DETECTION_MODEL_DIR')
            if env_path:
                path = Path(env_path)
            else:
                # Try relative path
                path = Path('./models')
        
        return path.resolve()
    
    def _initialize_mediapipe_models(self):
        """Initialize all MediaPipe models with validation"""
        
        # Face Detection
        face_detector_path = self.model_dir / "blaze_face_short_range.tflite"
        if not face_detector_path.exists():
            raise FileNotFoundError(f"Face detector model not found: {face_detector_path}")
        
        face_detector_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_detector_path)),
            min_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE
        )
        self.face_detector = vision.FaceDetector.create_from_options(face_detector_options)
        
        # Face Landmarker (Face Mesh)
        face_landmarker_path = self.model_dir / "face_landmarker.task"
        if not face_landmarker_path.exists():
            raise FileNotFoundError(f"Face landmarker model not found: {face_landmarker_path}")
        
        face_landmarker_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_landmarker_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=self.config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(face_landmarker_options)
        
        # Hand Landmarker
        hand_landmarker_path = self.model_dir / "hand_landmarker.task"
        if not hand_landmarker_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found: {hand_landmarker_path}")
        
        hand_landmarker_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_landmarker_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=self.config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_landmarker_options)
        
        # Pose Landmarker
        pose_landmarker_path = self.model_dir / "pose_landmarker_lite.task"
        if not pose_landmarker_path.exists():
            raise FileNotFoundError(f"Pose landmarker model not found: {pose_landmarker_path}")
        
        pose_landmarker_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_landmarker_path)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.config.MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=self.config.MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=self.config.MIN_TRACKING_CONFIDENCE
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_landmarker_options)
    
    def _initialize_yolo_model(self):
        """Initialize YOLO model for phone detection"""
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        yolo_path = self.model_dir / "yolov8n.pt"
        if not yolo_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
        
        self.phone_model = YOLO(str(yolo_path))
    
    def __del__(self):
        """Cleanup resources"""
        try:
            # MediaPipe models don't have explicit close, but we can nullify references
            self.face_detector = None
            self.face_mesh = None
            self.hands = None
            self.pose = None
            self.phone_model = None
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def extract_features(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract all features from an image.
        
        Args:
            image: Input image as BGR numpy array (H x W x 3)
           
        Returns:
            Dictionary containing 37 extracted features
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If extraction fails
        """
        try:
            # Validate input
            self._validate_image(image)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize features with defaults
            features = self._get_default_features()
            
            # Extract features in order
            face_features = self._extract_face_features(rgb_image, width, height)
            features.update(face_features)
            
            hand_features = self._extract_hand_features(rgb_image, width, height)
            features.update(hand_features)
            
            head_pose = self._extract_head_pose(rgb_image, width, height)
            features.update(head_pose)
            
            gaze_features = self._extract_gaze_features(rgb_image, width, height)
            features.update(gaze_features)
            
            phone_features = self._detect_phone(image)
            features.update(phone_features)
            
            logger.info(f"✓ Successfully extracted features from {width}x{height} image")
            return features
            
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            raise
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction error: {e}") from e
    
    @staticmethod
    def _validate_image(image: np.ndarray):
        """Validate image format and content"""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array")
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError(f"Invalid image dtype: {image.dtype}. Expected uint8 or float")
        
        if len(image.shape) != 3 or image.shape[2] not in [3, 4]:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected (H, W, 3) or (H, W, 4)")
        
        if image.size == 0:
            raise ValueError("Image is empty")
        
        min_dim = min(image.shape[:2])
        if min_dim < 50:
            raise ValueError(f"Image too small: {min_dim}px. Minimum 50px required")
    
    # ========================================================================
    # FEATURE EXTRACTION METHODS
    # ========================================================================
    
    def _get_default_features(self) -> Dict[str, any]:
        """Return default feature values for all 37 features"""
        return {
            # Face features (6)
            'face_present': 0,
            'no_of_face': 0,
            'face_x': 0.0,
            'face_y': 0.0,
            'face_w': 0.0,
            'face_h': 0.0,
            'face_conf': 0.0,
            
            # Eye features (8)
            'left_eye_x': 0.0,
            'left_eye_y': 0.0,
            'right_eye_x': 0.0,
            'right_eye_y': 0.0,
            'pupil_left_x': 0.0,
            'pupil_left_y': 0.0,
            'pupil_right_x': 0.0,
            'pupil_right_y': 0.0,
            
            # Face landmarks (4)
            'nose_tip_x': 0.0,
            'nose_tip_y': 0.0,
            'mouth_x': 0.0,
            'mouth_y': 0.0,
            
            # Hand features (5)
            'hand_count': 0,
            'left_hand_x': 0.0,
            'left_hand_y': 0.0,
            'right_hand_x': 0.0,
            'right_hand_y': 0.0,
            'hand_obj_interaction': 0,
            
            # Head pose (4)
            'head_pose': 'None',
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
            
            # Gaze features (4)
            'gaze_on_script': 0,
            'gaze_direction': 'None',
            'gazePoint_x': 0.0,
            'gazePoint_y': 0.0,
            
            # Phone detection (3)
            'phone_present': 0,
            'phone_loc_x': 0.0,
            'phone_loc_y': 0.0,
            'phone_conf': 0.0,
        }
    
    def _extract_face_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract face detection and mesh landmarks"""
        features = {}
        
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Face detection
            face_results = self.face_detector.detect(mp_image)
            
            if not face_results.detections:
                return features
            
            # Process first face
            detection = face_results.detections[0]
            bbox = detection.bounding_box
            
            features['face_present'] = 1
            features['no_of_face'] = len(face_results.detections)
            features['face_x'] = float(bbox.origin_x)
            features['face_y'] = float(bbox.origin_y)
            features['face_w'] = float(bbox.width)
            features['face_h'] = float(bbox.height)
            features['face_conf'] = float(detection.categories[0].score * 100)
            
            # Face mesh landmarks
            mesh_results = self.face_mesh.detect(mp_image)
            
            if not mesh_results.face_landmarks:
                return features
            
            landmarks = mesh_results.face_landmarks[0]
            
            # Eye centers
            left_eye_center = self._get_landmark_center(landmarks, MediaPipeLandmarks.LEFT_EYE, width, height)
            features['left_eye_x'] = float(left_eye_center[0])
            features['left_eye_y'] = float(left_eye_center[1])
            
            right_eye_center = self._get_landmark_center(landmarks, MediaPipeLandmarks.RIGHT_EYE, width, height)
            features['right_eye_x'] = float(right_eye_center[0])
            features['right_eye_y'] = float(right_eye_center[1])
            
            # Nose tip
            nose_tip = landmarks[MediaPipeLandmarks.NOSE_TIP]
            features['nose_tip_x'] = float(nose_tip.x * width)
            features['nose_tip_y'] = float(nose_tip.y * height)
            
            # Mouth center
            mouth_center = self._get_landmark_center(landmarks, MediaPipeLandmarks.MOUTH_LANDMARKS, width, height)
            features['mouth_x'] = float(mouth_center[0])
            features['mouth_y'] = float(mouth_center[1])
            
        except Exception as e:
            logger.warning(f"Error extracting face features: {e}")
        
        return features
    
    def _extract_hand_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract hand detection and landmarks"""
        features = {}
        
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            hand_results = self.hands.detect(mp_image)
            
            if not hand_results.hand_landmarks:
                features['hand_count'] = 0
                return features
            
            features['hand_count'] = len(hand_results.hand_landmarks)
            
            # Process each detected hand
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                hand_type = hand_results.handedness[idx].classification[0].label
                
                # Wrist position (landmark 0)
                wrist = hand_landmarks[0]
                wrist_x = float(wrist.x * width)
                wrist_y = float(wrist.y * height)
                
                if hand_type == 'Left':
                    features['left_hand_x'] = wrist_x
                    features['left_hand_y'] = wrist_y
                elif hand_type == 'Right':
                    features['right_hand_x'] = wrist_x
                    features['right_hand_y'] = wrist_y
            
            # Check hand-face interaction
            features['hand_obj_interaction'] = self._check_hand_face_interaction(features)
            
        except Exception as e:
            logger.warning(f"Error extracting hand features: {e}")
            features['hand_count'] = 0
        
        return features
    
    def _check_hand_face_interaction(self, features: Dict) -> int:
        """Check if hands are near face"""
        if features.get('face_present', 0) != 1:
            return 0
        
        face_y = features.get('face_y', 0)
        face_h = features.get('face_h', 0)
        
        if face_h == 0:
            return 0
        
        face_center_y = face_y + face_h / 2
        threshold = face_h * self.config.HAND_FACE_PROXIMITY_THRESHOLD
        
        left_hand_y = features.get('left_hand_y', 0)
        right_hand_y = features.get('right_hand_y', 0)
        
        if left_hand_y > 0 and abs(left_hand_y - face_center_y) < threshold:
            return 1
        if right_hand_y > 0 and abs(right_hand_y - face_center_y) < threshold:
            return 1
        
        return 0
    def _extract_head_pose(self, rgb_image: np.ndarray,  width: int, height: int) -> Dict:
        """Extract head pose (pitch, yaw, roll) using solvePnP"""
        features = {}
        
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.face_mesh.detect(mp_image)
            
            if not result.face_landmarks:
                return features

            landmarks = result.face_landmarks[0]

            # Get image points from landmarks
            image_points = self._get_image_points_for_head_pose(landmarks, width, height)

            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                HeadPosePoints.MODEL_POINTS,
                image_points,
                self._get_camera_matrix(width, height),
                np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                logger.warning("solvePnP failed for head pose")
                return features
            
            # Convert rotation vector to Euler angles
            rmat, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)
            
            features['head_pitch'] = float(pitch)
            features['head_yaw'] = float(yaw)
            features['head_roll'] = float(roll)
            
            # Determine head pose label
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            
            if abs(yaw_deg) > self.config.HEAD_POSE_THRESHOLD:
                features['head_pose'] = 'right' if yaw_deg > 0 else 'left'
            elif abs(pitch_deg) > self.config.HEAD_POSE_THRESHOLD:
                features['head_pose'] = 'down' if pitch_deg > 0 else 'up'
            else:
                features['head_pose'] = 'forward'
            
        except Exception as e:
            logger.warning(f"Error extracting head pose: {e}")
        
        return features
    
    def _extract_gaze_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract gaze direction and attention"""
        features = {
            'gaze_on_script': 0,
            'gaze_direction': 'None',
            'gazePoint_x': 0.0,
            'gazePoint_y': 0.0,
            'pupil_left_x': 0.0,
            'pupil_left_y': 0.0,
            'pupil_right_x': 0.0,
            'pupil_right_y': 0.0,
        }
        
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.face_mesh.detect(mp_image)
            
            if not result.face_landmarks:
                return features
            
            face_landmarks = result.face_landmarks[0]
            
            # Extract iris positions
            left_iris = self._get_iris_center(face_landmarks, MediaPipeLandmarks.LEFT_IRIS, width, height)
            right_iris = self._get_iris_center(face_landmarks, MediaPipeLandmarks.RIGHT_IRIS, width, height)
            
            features['pupil_left_x'] = left_iris[0]
            features['pupil_left_y'] = left_iris[1]
            features['pupil_right_x'] = right_iris[0]
            features['pupil_right_y'] = right_iris[1]
            
            # Gaze point (average of both irises)
            gaze_x = (left_iris[0] + right_iris[0]) / 2
            gaze_y = (left_iris[1] + right_iris[1]) / 2
            features['gazePoint_x'] = gaze_x
            features['gazePoint_y'] = gaze_y
            
            # Estimate gaze direction using head pose
            head_pose_features = self._extract_head_pose(rgb_image, width, height)
            
            if head_pose_features:
                gaze_direction = self._classify_gaze_direction(
                    gaze_x, gaze_y, width, height,
                    head_pose_features
                )
                features['gaze_direction'] = gaze_direction
                
                # Determine if gaze is on script
                features['gaze_on_script'] = 1 if gaze_direction == 'down' or gaze_y > height * self.config.SCRIPT_REGION_Y_THRESHOLD else 0
            
        except Exception as e:
            logger.warning(f"Error extracting gaze features: {e}")
        
        return features
    
    def _detect_phone(self, image: np.ndarray) -> Dict:
        """Detect phone in image using YOLO"""
        features = {
            'phone_present': 0,
            'phone_loc_x': 0.0,
            'phone_loc_y': 0.0,
            'phone_conf': 0.0,
        }
        
        if self.phone_model is None:
            return features
        
        try:
            results = self.phone_model(image, verbose=False)
            
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    # COCO class 67 = cellphone
                    if cls == 67:
                        x1, y1, x2, y2 = box.xyxy[0]
                        features['phone_present'] = 1
                        features['phone_loc_x'] = float((x1 + x2) / 2)
                        features['phone_loc_y'] = float((y1 + y2) / 2)
                        features['phone_conf'] = float(box.conf[0].item())
                        return features  # Return on first detected phone
        
        except Exception as e:
            logger.warning(f"Error detecting phone: {e}")
        
        return features
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @staticmethod
    def _get_landmark_center(
        landmarks: List,
        landmark_indices: List[int],
        width: int,
        height: int
    ) -> Tuple[float, float]:
        """Calculate center point of specified landmarks"""
        try:
            x_coords = [landmarks[i].x * width for i in landmark_indices if i < len(landmarks)]
            y_coords = [landmarks[i].y * height for i in landmark_indices if i < len(landmarks)]
            
            if not x_coords or not y_coords:
                return 0.0, 0.0
            
            return float(np.mean(x_coords)), float(np.mean(y_coords))
        except Exception as e:
            logger.warning(f"Error calculating landmark center: {e}")
            return 0.0, 0.0
    
    @staticmethod
    def _get_iris_center(
        landmarks: List,
        iris_indices: List[int],
        width: int,
        height: int
    ) -> Tuple[float, float]:
        """Calculate center of iris/pupil"""
        try:
            xs = [landmarks[i].x * width for i in iris_indices if i < len(landmarks)]
            ys = [landmarks[i].y * height for i in iris_indices if i < len(landmarks)]
            
            if not xs or not ys:
                return 0.0, 0.0
            
            return float(np.mean(xs)), float(np.mean(ys))
        except Exception as e:
            logger.warning(f"Error calculating iris center: {e}")
            return 0.0, 0.0
    
    def _get_image_points_for_head_pose(
        self,
        landmarks: List,
        width: int,
        height: int
    ) -> np.ndarray:
        """Get image points for head pose estimation"""
        image_points = []
        for idx in HeadPosePoints.LANDMARK_INDICES:
            if idx < len(landmarks):
                image_points.append([
                    landmarks[idx].x * width,
                    landmarks[idx].y * height
                ])
        
        return np.array(image_points, dtype=np.float64)
    
    @staticmethod
    def _get_camera_matrix(width: int, height: int) -> np.ndarray:
        """Create camera intrinsic matrix"""
        focal_length = width
        center_x = width / 2
        center_y = height / 2
        
        return np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float64)
    
    @staticmethod
    def _rotation_matrix_to_euler(rmat: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = 0
        
        return pitch, yaw, roll
    
    def _classify_gaze_direction(
        self,
        gaze_x: float,
        gaze_y: float,
        width: int,
        height: int,
        head_pose_features: Dict
    ) -> str:
        """Classify gaze direction based on eye position"""
        center_x = width / 2
        center_y = height / 2
        
        dx = gaze_x - center_x
        dy = gaze_y - center_y
        
        threshold = self.config.GAZE_DIRECTION_THRESHOLD
        center_threshold = self.config.GAZE_CENTER_THRESHOLD
        
        if abs(dx) < center_threshold and abs(dy) < center_threshold:
            return 'center'
        elif dy > threshold:
            return 'down'
        elif dy < -threshold:
            return 'up'
        elif dx > threshold:
            return 'right'
        else:
            return 'left'

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def draw_annotations(self, image: np.ndarray, features: Dict) -> np.ndarray:
        """
        Draw annotations on image for visualization.
        
        Args:
            image: Input image (BGR)
            features: Feature dictionary
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        try:
            # Face
            if features.get('face_present', 0) == 1:
                self._draw_face_box(annotated, features)
            
            # Eyes
            self._draw_eyes(annotated, features)
            
            # Hands
            self._draw_hands(annotated, features)
            
            # Gaze
            self._draw_gaze(annotated, features, width, height)
            
            # Phone
            if features.get('phone_present', 0) == 1:
                self._draw_phone(annotated, features, width, height)
            
            # Head pose
            cv2.putText(
                annotated,
                f"Head: {features.get('head_pose', 'None')}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        except Exception as e:
            logger.warning(f"Error drawing annotations: {e}")
        
        return annotated
    
    @staticmethod
    def _draw_face_box(image: np.ndarray, features: Dict):
        """Draw face bounding box"""
        x, y, w, h = int(features['face_x']), int(features['face_y']), int(features['face_w']), int(features['face_h'])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Face: {features['face_conf']:.1f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    @staticmethod
    def _draw_eyes(image: np.ndarray, features: Dict):
        """Draw eye positions"""
        if features.get('left_eye_x', 0) > 0:
            cv2.circle(image, (int(features['left_eye_x']), int(features['left_eye_y'])), 5, (255, 0, 0), -1)
        if features.get('right_eye_x', 0) > 0:
            cv2.circle(image, (int(features['right_eye_x']), int(features['right_eye_y'])), 5, (255, 0, 0), -1)
    
    @staticmethod
    def _draw_hands(image: np.ndarray, features: Dict):
        """Draw hand positions"""
        if features.get('left_hand_x', 0) > 0:
            cv2.circle(image, (int(features['left_hand_x']), int(features['left_hand_y'])), 10, (0, 0, 255), -1)
        if features.get('right_hand_x', 0) > 0:
            cv2.circle(image, (int(features['right_hand_x']), int(features['right_hand_y'])), 10, (0, 0, 255), -1)
    
    @staticmethod
    def _draw_gaze(image: np.ndarray, features: Dict, width: int, height: int):
        """Draw gaze point and direction"""
        if features.get('gazePoint_x', 0) > 0:
            cv2.circle(image, (int(features['gazePoint_x']), int(features['gazePoint_y'])), 8, (255, 255, 0), -1)
            cv2.putText(
                image,
                f"Gaze: {features['gaze_direction']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
    
    @staticmethod
    def _draw_phone(image: np.ndarray, features: Dict, width: int, height: int):
        """Draw detected phone"""
        cv2.circle(image, (int(features['phone_loc_x']), int(features['phone_loc_y'])), 15, (0, 0, 255), 2)
        cv2.putText(
            image,
            f"Phone: {features['phone_conf']:.1f}%",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_feature_extractor(
    model_dir: Optional[str] = None,
    use_phone_detection: bool = True
) -> FeatureExtractor:
    """
    Factory function to create a FeatureExtractor with proper error handling.
    
    Args:
        model_dir: Path to models directory
        use_phone_detection: Enable phone detection
        
    Returns:
        Initialized FeatureExtractor
        
    Raises:
        FileNotFoundError: If models not found
        RuntimeError: If initialization fails
    """
    try:
        extractor = FeatureExtractor(
            model_dir=model_dir,
            use_phone_detection=use_phone_detection
        )
        logger.info("✓ Feature extractor created successfully")
        return extractor
    except Exception as e:
        logger.error(f"✗ Failed to create feature extractor: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    logger.info("Feature Extraction Module Ready")
    logger.info(f"Default features: {len(FeatureExtractor(model_dir='./models')._get_default_features())} total")