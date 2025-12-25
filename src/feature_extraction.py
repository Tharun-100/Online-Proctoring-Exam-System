"""
Feature Extraction Module for Fraud Detection System
Extracts 37 features from input images for fraud detection
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
import math


class FeatureExtractor:
    """Extract features from exam proctoring images"""
    
    def __init__(self):
        """Initialize MediaPipe models for feature extraction"""
        # MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # MediaPipe face mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # MediaPipe pose for head pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        
        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
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
        head_pose = self._extract_head_pose(rgb_image, face_features, width, height)
        features.update(head_pose)
        
        # Extract gaze features
        gaze_features = self._extract_gaze_features(face_features, width, height)
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
        face_results = self.face_detection.process(rgb_image)
        
        if face_results.detections:
            # Get the first face (assuming single person)
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            features['face_present'] = 1
            features['no_of_face'] = len(face_results.detections)
            features['face_x'] = bbox.xmin * width
            features['face_y'] = bbox.ymin * height
            features['face_w'] = bbox.width * width
            features['face_h'] = bbox.height * height
            features['face_conf'] = detection.score[0] * 100
            
            # Face mesh for detailed landmarks
            mesh_results = self.face_mesh.process(rgb_image)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]
                
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
                nose_tip = landmarks.landmark[4]
                features['nose_tip_x'] = nose_tip.x * width
                features['nose_tip_y'] = nose_tip.y * height
                
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
        
        hand_results = self.hands.process(rgb_image)
        
        if hand_results.multi_hand_landmarks:
            features['hand_count'] = len(hand_results.multi_hand_landmarks)
            
            # Get hand landmarks
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Get hand type (left or right)
                hand_type = hand_results.multi_handedness[idx].classification[0].label
                
                # Get wrist position (landmark 0)
                wrist = hand_landmarks.landmark[0]
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
    
    def _extract_head_pose(self, rgb_image: np.ndarray, face_features: Dict, width: int, height: int) -> Dict:
        """Extract head pose (pitch, yaw, roll)"""
        features = {}
        
        if face_features.get('face_present', 0) == 1:
            # Use face landmarks to estimate head pose
            mesh_results = self.face_mesh.process(rgb_image)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]
                
                # Key facial points for pose estimation
                # Nose tip, chin, left eye, right eye, left mouth corner, right mouth corner
                nose_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y])
                chin = np.array([landmarks.landmark[152].x, landmarks.landmark[152].y])
                left_eye = np.array([landmarks.landmark[33].x, landmarks.landmark[33].y])
                right_eye = np.array([landmarks.landmark[362].x, landmarks.landmark[362].y])
                left_mouth = np.array([landmarks.landmark[61].x, landmarks.landmark[61].y])
                right_mouth = np.array([landmarks.landmark[291].x, landmarks.landmark[291].y])
                
                # Calculate pitch (vertical rotation)
                nose_chin_vec = chin - nose_tip
                pitch = math.atan2(nose_chin_vec[1], abs(nose_chin_vec[0])) if abs(nose_chin_vec[0]) > 0 else 0
                features['head_pitch'] = float(pitch)
                
                # Calculate yaw (horizontal rotation)
                eye_center = (left_eye + right_eye) / 2
                mouth_center = (left_mouth + right_mouth) / 2
                face_center = (eye_center + mouth_center) / 2
                yaw_offset = face_center[0] - 0.5  # Offset from image center
                features['head_yaw'] = float(yaw_offset * 0.5)  # Scale factor
                
                # Calculate roll (tilt)
                eye_vec = right_eye - left_eye
                roll = math.atan2(eye_vec[1], eye_vec[0])
                features['head_roll'] = float(roll)
                
                # Determine head pose direction
                if abs(pitch) > 0.1:
                    features['head_pose'] = 'down' if pitch > 0 else 'up'
                elif abs(yaw_offset) > 0.15:
                    features['head_pose'] = 'right' if yaw_offset > 0 else 'left'
                else:
                    features['head_pose'] = 'forward'
            else:
                features['head_pose'] = 'None'
        else:
            features['head_pose'] = 'None'
        
        return features
    
    def _extract_gaze_features(self, face_features: Dict, width: int, height: int) -> Dict:
        """Extract gaze-related features"""
        features = {}
        
        if face_features.get('face_present', 0) == 1:
            # Calculate gaze point (simplified: center of eyes)
            left_eye_x = face_features.get('left_eye_x', 0)
            left_eye_y = face_features.get('left_eye_y', 0)
            right_eye_x = face_features.get('right_eye_x', 0)
            right_eye_y = face_features.get('right_eye_y', 0)
            
            if left_eye_x > 0 and right_eye_x > 0:
                gaze_x = (left_eye_x + right_eye_x) / 2
                gaze_y = (left_eye_y + right_eye_y) / 2
                
                features['gazePoint_x'] = gaze_x
                features['gazePoint_y'] = gaze_y
                
                # Determine gaze direction
                center_x = width / 2
                center_y = height / 2
                
                offset_x = gaze_x - center_x
                offset_y = gaze_y - center_y
                
                threshold = 50  # pixels
                
                if abs(offset_x) < threshold and abs(offset_y) < threshold:
                    features['gaze_direction'] = 'center'
                elif abs(offset_x) > abs(offset_y):
                    features['gaze_direction'] = 'right' if offset_x > 0 else 'left'
                else:
                    features['gaze_direction'] = 'bottom_right' if offset_y > 0 else 'top_left'
                
                # Check if gaze is on script (heuristic: looking down or bottom area)
                if features['gaze_direction'] in ['bottom_right', 'bottom_left'] or gaze_y > height * 0.6:
                    features['gaze_on_script'] = 1
                else:
                    features['gaze_on_script'] = 0
            else:
                features['gaze_direction'] = 'None'
        else:
            features['gaze_direction'] = 'None'
        
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
        
        # Simple heuristic: detect rectangular objects that might be phones
        # This is a simplified version - real implementation would use object detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 50000:  # Phone-like size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0:  # Phone-like aspect ratio
                    features['phone_present'] = 1
                    features['phone_loc_x'] = x + w / 2
                    features['phone_loc_y'] = y + h / 2
                    features['phone_conf'] = 0.5  # Placeholder confidence
                    break
        
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



