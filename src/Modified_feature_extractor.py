"""
Feature Extraction Module for Fraud Detection System
Extracts 37 features from input images for fraud detection
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
import math
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_dir="models"
class FeatureExtractor:
    def __init__(self, model_dir: str = "models"):
        """
        Initialize MediaPipe Tasks for feature extraction
        (Face detection, face mesh, hands, pose)

        model_dir:
            Directory containing .task models
        """

        # ---------- Face Detection ----------
        face_detector_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=f"{model_dir}/blaze_face_short_range.tflite"
            ),
            min_detection_confidence=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(
            face_detector_options
        )
        # ---------- Face Mesh (Face Landmarker) ----------
        face_landmarker_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=f"{model_dir}/face_landmarker.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(
            face_landmarker_options
        )

        # ---------- Hands ----------
        hand_landmarker_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=f"{model_dir}/hand_landmarker.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = vision.HandLandmarker.create_from_options(
            hand_landmarker_options
        )
        
        # ---------- Pose ----------
        pose_landmarker_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=f"{model_dir}/pose_landmarker_lite.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = vision.PoseLandmarker.create_from_options(
            pose_landmarker_options
        )
        # ---------- Phone Detector (YOLO) ----------
        self.phone_model = YOLO(f"{model_dir}/yolov8n.pt")  # COCO class 67 = cellphone

    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract all features from an image
        
        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Dictionary with extracted features
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        # Initialize default feature values
        features = self._get_default_features()

        # Extract face features
        face_features = self._extract_face_features(rgb_image, width, height)
        features.update(face_features)
        
        # Extract hand features
        hand_features = self._extract_hand_features(rgb_image, width, height)
        features.update(hand_features)
        
        # Extract head pose
        head_pose = self._extract_head_pose(rgb_image, width, height)
        features.update(head_pose)

        # Extract gaze features
        gaze_features = self._extract_gaze_features(rgb_image, width, height)
        features.update(gaze_features)
        
        # Detect phone (placeholder - would need object detection model)
        phone_features = self._detect_phone(image)
        features.update(phone_features)

        return features
    
    def _get_default_features(self) -> Dict:
        """Return default feature values"""
        return {
            'face_present': 0,
            'no_of_face': 0,
            'face_x': 0,
            'face_y': 0,
            'face_w': 0,
            'face_h': 0,
            'left_eye_x': 0,
            'left_eye_y': 0,
            'right_eye_x': 0,
            'right_eye_y': 0,
            'nose_tip_x': 0,
            'nose_tip_y': 0,
            'mouth_x': 0,
            'mouth_y': 0,
            'face_conf': 0,
            'hand_count': 0,
            'left_hand_x': 0,
            'left_hand_y': 0,
            'right_hand_x': 0,
            'right_hand_y': 0,
            'hand_obj_interaction': 0,
            'head_pose': 'None',
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
            'phone_present': 0,
            'phone_loc_x': 0,
            'phone_loc_y': 0,
            'phone_conf': 0,
            'gaze_on_script': 0,
            'gaze_direction': 'None',
            'gazePoint_x': 0,
            'gazePoint_y': 0,
            'pupil_left_x': 0,
            'pupil_left_y': 0,
            'pupil_right_x': 0,
            'pupil_right_y': 0
        }
    
    def _extract_face_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract face-related features"""
        features = {}
        # Face detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        face_results = self.face_detector.detect(mp_image)
        # mediapipe image is done:-

        if face_results.detections:
            # Get the first face (assuming single person)
            detection = face_results.detections[0]
            bbox = detection.bounding_box
            features['face_present'] = 1
            features['no_of_face'] = len(face_results.detections)
            features['face_x'] = bbox.origin_x 
            features['face_y'] = bbox.origin_y
            features['face_w'] = bbox.width
            features['face_h'] = bbox.height
            features['face_conf'] = detection.categories[0].score*100
            #-------------- Face mesh landmarks--------------

            mesh_results = self.face_mesh.detect(rgb_image)

            if mesh_results.face_landmarks:
                landmarks = mesh_results.face_landmarks[0]
                # Left eye (MediaPipe landmarks: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
                left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                left_eye_center = self._get_landmark_center(landmarks, left_eye_landmarks, width, height)
                features['left_eye_x'] = left_eye_center[0]
                features['left_eye_y'] = left_eye_center[1]
                features['pupil_left_x'] = left_eye_center[0]
                features['pupil_left_y'] = left_eye_center[1]
                
                # Right eye (MediaPipe landmarks: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
                right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                right_eye_center = self._get_landmark_center(landmarks, right_eye_landmarks, width, height)
                features['right_eye_x'] = right_eye_center[0]
                features['right_eye_y'] = right_eye_center[1]
                features['pupil_right_x'] = right_eye_center[0]
                features['pupil_right_y'] = right_eye_center[1]
                
                # Nose tip (landmark 4)

                nose_tip = landmarks[4]
                features['nose_tip_x'] = int(nose_tip.x * width)
                features['nose_tip_y'] = int(nose_tip.y * height)
                
                # Mouth center (landmarks: 13, 14, 308, 324, 318)
                mouth_landmarks = [13, 14, 308, 324, 318]
                mouth_center = self._get_landmark_center(landmarks, mouth_landmarks, width, height)
                features['mouth_x'] = mouth_center[0]
                features['mouth_y'] = mouth_center[1]
        else:
            features['face_present'] = 0
            features['no_of_face'] = 0
            
        return features
    
    def _extract_hand_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract hand-related features"""
        features = {}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        hand_results = self.hands.detect(mp_image)

        if hand_results.hand_landmarks:
            features['hand_count'] = len(hand_results.hand_landmarks)
            
            # Get hand landmarks
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                # Get hand type (left or right)
                hand_type = hand_results.handedness[idx].classification[0].label
                
                # Get wrist position (landmark 0)
                wrist = hand_landmarks[0]
                wrist_x = wrist.x * width
                wrist_y = wrist.y * height
                
                if hand_type == 'Left':
                    features['left_hand_x'] = wrist_x
                    features['left_hand_y'] = wrist_y
                elif hand_type == 'Right':
                    features['right_hand_x'] = wrist_x
                    features['right_hand_y'] = wrist_y
            
            # Check if hands are near face (potential interaction)
            if 'face_y' in features and features['face_present'] == 1:
                face_y = features.get('face_y', 0)
                face_h = features.get('face_h', 0)
                face_center_y = face_y + face_h / 2
                
                # Simple heuristic: if hand is near face region
                hand_near_face = False
                if features.get('left_hand_y', 0) > 0:
                    if abs(features['left_hand_y'] - face_center_y) < face_h * 1.5:
                        hand_near_face = True
                if features.get('right_hand_y', 0) > 0:
                    if abs(features['right_hand_y'] - face_center_y) < face_h * 1.5:
                        hand_near_face = True

                features['hand_obj_interaction'] = 1 if hand_near_face else 0
        else:
            features['hand_count'] = 0
        return features
    
    def _extract_head_pose(self, rgb_image: np.ndarray,  width: int, height: int) -> Dict:
        """Extract head pose (pitch, yaw, roll)"""
        features = {}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.face_mesh.detect(mp_image)
        if not result.face_landmarks:
            return features
        landmarks = result.face_landmarks[0]
        image_points = np.array([
        (landmarks[1].x * width,   landmarks[1].y * height),   # Nose tip
        (landmarks[152].x * width, landmarks[152].y * height), # Chin
        (landmarks[33].x * width,  landmarks[33].y * height),  # Left eye
        (landmarks[263].x * width, landmarks[263].y * height), # Right eye
        (landmarks[61].x * width,  landmarks[61].y * height),  # Left mouth
        (landmarks[291].x * width, landmarks[291].y * height)  ], dtype=np.float64)

        model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -63.6, -12.5),   # Chin
        (-43.3, 32.7, -26.0),  # Left eye
        (43.3, 32.7, -26.0),   # Right eye
        (-28.9, -28.9, -24.1),# Left mouth
        (28.9, -28.9, -24.1) ], dtype=np.float64)
        focal_length = width
        center = (width/2, height/2)
        camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return features
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw   = np.arctan2(-rmat[2, 0], sy)
            roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw   = np.arctan2(-rmat[2, 0], sy)
            roll  = 0
        features["head_pitch"] = np.degrees(pitch)
        features["head_yaw"]   = np.degrees(yaw)
        features["head_roll"]  = np.degrees(roll)
            # Head pose label
            
        pitch=np.degrees(pitch)
        yaw=np.degrees(yaw)
        roll=np.degrees(roll)
    # -----------------------------

        if abs(yaw) > 20:
            features["head_pose"] = "right" if yaw > 0 else "left"
        elif abs(pitch) > 20:
            features["head_pose"] = "down" if pitch > 0 else "up"
        else:
            features["head_pose"] = "forward"
        return features

    def _extract_gaze_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract gaze-related features"""
        features = {'gaze_on_script': 0,'gaze_direction': 'None','gazePoint_x': 0,'gazePoint_y': 0,'pupil_left_x': 0,'pupil_left_y': 0,'pupil_right_x': 0,'pupil_right_y': 0}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_image)
        result = self.face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            features['gaze_on_script'] = 0
            features['gaze_direction'] = 'None'
            return features
        face_landmarks = result.face_landmarks[0]
        # Iris Landmarks:
        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]
        def iris_center(ids):
            xs = [face_landmarks[i].x * width for i in ids]
            ys = [face_landmarks[i].y * height for i in ids]
            return float(np.mean(xs)), float(np.mean(ys))
        left_iris_x, left_iris_y = iris_center(LEFT_IRIS)
        right_iris_x, right_iris_y = iris_center(RIGHT_IRIS)
        features['pupil_left_x'] = left_iris_x
        features['pupil_left_y'] = left_iris_y
        features['pupil_right_x'] = right_iris_x
        features['pupil_right_y'] = right_iris_y
        # 2. Eye center (average pupil)
        gaze_point_x = (left_iris_x + right_iris_x) / 2
        gaze_point_y = (left_iris_y + right_iris_y) / 2
        features['gazePoint_x'] = gaze_point_x
        features['gazePoint_y'] = gaze_point_y
        # Head Pose Estimation using solve
        model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -63.6, -12.5),   # Chin
        (-43.3, 32.7, -26.0),  # Left eye corner
        (43.3, 32.7, -26.0),   # Right eye corner
        (-28.9, -28.9, -24.1), # Left mouth
        (28.9, -28.9, -24.1)   # Right mouth
        ], dtype=np.float64)

        image_points = np.array([
        (face_landmarks[4].x * width,   face_landmarks[4].y * height),    # Nose tip
        (face_landmarks[152].x * width, face_landmarks[152].y * height),  # Chin
        (face_landmarks[33].x * width,  face_landmarks[33].y * height),   # Left eye corner
        (face_landmarks[263].x * width, face_landmarks[263].y * height),  # Right eye corner
        (face_landmarks[61].x * width,  face_landmarks[61].y * height),   # Left mouth
        (face_landmarks[291].x * width, face_landmarks[291].y * height)   # Right mouth
        ], dtype=np.float64)

        focal_length = width
        center=(width/2, height/2)
        camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=np.float64)

        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

        success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            features['gaze_on_script'] = 0
            features['gaze_direction'] = 'None'
            return features
        
        # Project gaze direction 
        gaze_3d= np.array([[0,0,1000.0]])
        gaze_2d, _ = cv2.projectPoints(gaze_3d, rvec, tvec, camera_matrix, dist_coeffs)

        gaze_2d_x,gaze_2d_y = gaze_2d[0][0]
        # gaze direction classification
        dx=gaze_2d_x - center[0]
        dy=gaze_2d_y - center[1]

        if abs(dx)<80 and abs(dy)<80:
            gaze_direction='center'
        elif dy >100 :
            gaze_direction='down'
        elif dy < -100:
            gaze_direction='up'
        elif dx >100:
            gaze_direction='right'
        else :
            gaze_direction='left'
        
        features['gaze_direction'] = gaze_direction

        # Gaze on script determination using simple heuristics

        if gaze_direction == 'down' or gaze_point_y > height*0.6:
            features['gaze_on_script'] = 1
        else:
            features['gaze_on_script'] = 0

        return features
    
    def _detect_phone(self, image: np.ndarray) -> Dict:
        """Detect phone in image (placeholder - would need YOLO or similar)"""
        # This is a placeholder - in production, you'd use an object detection model
        # For now, return default values
        features = {
            'phone_present': 0,
            'phone_loc_x': 0,
            'phone_loc_y': 0,
            'phone_conf': 0
        }
        # Using YOLO Models:
        results = self.phone_model(image)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 67:
                    features['phone_present'] = 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    features['phone_loc_x'] = int((x1 + x2) / 2)
                    features['phone_loc_y'] = int((y1 + y2) / 2)
                    features['phone_conf'] = float(box.conf[0])
                    return features # Return on first detected phone
        return features
    
    def _get_landmark_center(self, landmarks, landmark_indices: list, width: int, height: int) -> Tuple[float, float]:
        """Calculate center point of specified landmarks"""
        x_coords = [landmarks.landmark[i].x * width for i in landmark_indices]
        y_coords = [landmarks.landmark[i].y * height for i in landmark_indices]
        return (np.mean(x_coords), np.mean(y_coords))
    
    def draw_annotations(self, image: np.ndarray, features: Dict) -> np.ndarray:
        """Draw annotations on image for visualization"""
        annotated_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw face bounding box
        if features.get('face_present', 0) == 1:
            x = int(features['face_x'])
            y = int(features['face_y'])
            w = int(features['face_w'])
            h = int(features['face_h'])
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Face: {features['face_conf']:.1f}%", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw eye positions
        if features.get('left_eye_x', 0) > 0:
            cv2.circle(annotated_image, (int(features['left_eye_x']), int(features['left_eye_y'])), 
                      5, (255, 0, 0), -1)
        if features.get('right_eye_x', 0) > 0:
            cv2.circle(annotated_image, (int(features['right_eye_x']), int(features['right_eye_y'])), 
                      5, (255, 0, 0), -1)
        
        # Draw hand positions
        if features.get('left_hand_x', 0) > 0:
            cv2.circle(annotated_image, (int(features['left_hand_x']), int(features['left_hand_y'])), 
                      10, (0, 0, 255), -1)
        if features.get('right_hand_x', 0) > 0:
            cv2.circle(annotated_image, (int(features['right_hand_x']), int(features['right_hand_y'])), 
                      10, (0, 0, 255), -1)
        
        # Draw gaze point
        if features.get('gazePoint_x', 0) > 0:
            cv2.circle(annotated_image, (int(features['gazePoint_x']), int(features['gazePoint_y'])), 
                      8, (255, 255, 0), -1)
            cv2.putText(annotated_image, f"Gaze: {features['gaze_direction']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw phone if detected
        if features.get('phone_present', 0) == 1:
            cv2.circle(annotated_image, (int(features['phone_loc_x']), int(features['phone_loc_y'])), 
                      15, (0, 0, 255), 2)
            cv2.putText(annotated_image, "Phone Detected", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Add head pose info
        cv2.putText(annotated_image, f"Head Pose: {features['head_pose']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated_image