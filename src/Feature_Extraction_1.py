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
import os
from pathlib import Path

DEBUG = False
PHONE_CONF_TH = 0.5

class FeatureExtractor:
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize MediaPipe Tasks for feature extraction
        (Face detection, face mesh, hands, pose)
        model_dir:
            Directory containing .task models
        """
        if model_dir is None:
            model_dir = str(Path(__file__).resolve().parent / "models")

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Pass `model_dir=...` or ensure `src/models` exists."
            )
        # ---------- Face Detection ----------
        face_detector_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir ,"blaze_face_short_range.tflite")
            ),
            min_detection_confidence=0.5
        )

        self.face_detector = vision.FaceDetector.create_from_options(
            face_detector_options
        )

        # ---------- Face Mesh (Face Landmarker) ----------
        
        face_landmarker_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir, "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            # min_face_detection_confidence=0.5,
            # min_face_presence_confidence=0.5,
            # min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.face_mesh = vision.FaceLandmarker.create_from_options(
            face_landmarker_options
        )

        # ---------- Hands ----------
        hand_landmarker_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(model_dir, "hand_landmarker.task")
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
                model_asset_path=os.path.join(model_dir,"pose_landmarker_lite.task")
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
        phone_model_path = os.path.join(model_dir, "yolov9m.pt")
        self.phone_model = YOLO(phone_model_path if os.path.exists(phone_model_path) else "yolov9m")
        self.phone_features = {}

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

        # Detect phone (placeholder - would need object detection model)
        phone_features = self._detect_phone(image)
        features.update(phone_features)

        # Extract head pose
        head_pose = self._extract_head_pose(rgb_image, width, height)
        features.update(head_pose)

        # Extract gaze features
        gaze_features = self._extract_gaze_features(rgb_image, width, height)
        features.update(gaze_features)

        return features
    
    def _get_default_features(self) -> Dict:
        """Return default feature values"""
        
        return {
            'number_of_people': 0,
            'face_conf': 0,
            'head_pose': 0, # <---: by default, we are thinking that it is on the screen :---> 
            'head_pitch': 0.0,
            'head_yaw': 0.0,
            'head_roll': 0.0,
            'phone_present': 0,
            'phone_conf': 0.0,
            'gaze_on_script': 0,
            'gaze_direction': 0,
            'gazePoint_x': 0,
            'gazePoint_y': 0,
        }
    
    def _extract_face_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract face-related features"""
        features = {}
        # Face detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        face_results = self.face_detector.detect(mp_image)
        # mediapipe image is done:

        if face_results.detections:
            # Get the first face (assuming single person)
            detection = face_results.detections[0]
            bbox = detection.bounding_box
            # features['face_present'] = 1
            features['number_of_people'] = int(len(face_results.detections))
            if DEBUG:
                print("Number of people detected:", features['number_of_people'])
            
            # features['face_x'] = float(bbox.origin_x)
            # features['face_y'] = float(bbox.origin_y)
            # features['face_w'] = float(bbox.width)
            # features['face_h'] = float(bbox.height)
            
            features['face_conf'] = detection.categories[0].score * 100

            #-------------- Face mesh landmarks--------------
        else:
            features['number_of_people'] = 0
            features['face_conf'] = 0

        return features

    def _extract_head_pose(self, rgb_image: np.ndarray,  width: int, height: int) -> Dict:
        """Extract head pose (pitch, yaw, roll)"""
        # we need to make sure with the floating point operations

        focal_length = width
        center = (width/2, height/2)

        camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion:-   
        features = {"head_pitch":0.0,"head_yaw":0.0,"head_roll":0.0,"head_pose":"None"} #,"nose_2d":(0,0),"nose_3d":(0,0,0),"rvec": (0,0,0),"tvec": (0,0,0),"camera_matrix": camera_matrix,"dist_coeffs": dist_coeffs}

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

        # These are the trouble some points:

        model_points= np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),   # Chin
            (-43.3, 32.7, -26.0),  # Left eye left corner
            (43.3, 32.7, -26.0),   # Right eye right corner
            (-28.9, -28.9, -24.1), # Left mouth corner
            (28.9, -28.9, -24.1)   # Right mouth corner
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(model_points,  image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return features
        
        rmat, _ = cv2.Rodrigues(rvec)

        # Calculate Euler angles from rotation matrix:-

        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw   = euler_angles[1][0]
        roll  = euler_angles[2][0]

        # Normalization

        if yaw > 90 :
            yaw= 180-yaw

        elif yaw < -90:
            yaw=-(180+yaw)

        if roll > 90 :
            roll= 180-roll

        elif roll < -90:
            roll=-(180+roll)

        if pitch > 90 :
            pitch= 180-pitch

        elif pitch < -90:
            pitch=-(180+pitch)

        features["head_pitch"] = np.radians(pitch)
        features["head_yaw"]   = np.radians(yaw)
        features["head_roll"]  = np.radians(roll)

        # Head pose label
        print("Head Pose - Pitch:", pitch, "Yaw:", yaw, "Roll:", roll)

        # print(features["head_pitch"])
        # print(features["head_yaw"])
        # print(features["head_roll"])

    # -----------------------------In General pitch has more dominance over pitch:--------------------

        if abs(yaw) <= 20 and abs(pitch) <= 15:
            features["head_pose"] = 0 # On the screen
        else :
            features['head_pose']= 1 # away from the screen:--->  
        return features

    def _extract_gaze_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract gaze-related features"""
        features = {'gaze_direction': 0} 
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_image)
        result = self.face_mesh.detect(mp_image)

        if not result.face_landmarks:
            features['gaze_direction'] = 1
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
        
        # 1. Pupil positions
        # features['pupil_left_x'] = left_iris_x
        # features['pupil_left_y'] = left_iris_y
        # features['pupil_right_x'] = right_iris_x
        # features['pupil_right_y'] = right_iris_y
        # 2. Eye center (average pupil)

        gaze_point_x = (left_iris_x + right_iris_x) / 2
        gaze_point_y = (left_iris_y + right_iris_y) / 2

        features['gazePoint_x'] = int(gaze_point_x)
        features['gazePoint_y'] = int(gaze_point_y)
        # are they useful?
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
            # features['gaze_on_script'] = 0
            features['gaze_direction'] = 1
            return features

        gaze_3d= np.array([[0,0,1000.0]])
        gaze_2d, _ = cv2.projectPoints(gaze_3d, rvec, tvec, camera_matrix, dist_coeffs)

        gaze_2d_x,gaze_2d_y = gaze_2d[0][0]
        # gaze direction classification

        dx=gaze_2d_x - center[0]
        dy=gaze_2d_y - center[1]

        CENTER_TH_X = 0.08 * width    # ~8% of width
        CENTER_TH_Y = 0.08 * height   # ~8% of height
        DIR_TH_X = 0.12 * width       # directional threshold
        DIR_TH_Y = 0.12 * height

        if abs(dx) < CENTER_TH_X and abs(dy) < CENTER_TH_Y:
            gaze_direction = 0 # looking at the screen
        else:
            gaze_direction = 1 # looking away from the screen
        features['gaze_direction'] = gaze_direction

        return features
    
    def _detect_phone(self, image: np.ndarray) -> Dict:
        """Detect phone in image (placeholder - would need YOLO or similar)"""
        features = {
            "phone_present": 0,
            "phone_conf": 0.0,
        }

        phone_feats={}

        results = self.phone_model(image, conf=PHONE_CONF_TH, verbose=False)
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                cls = int(b.cls[0])
                if cls==67:  # COCO class ID for cell phone
                    conf= float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()  # Get bounding box coordinates
                    features["phone_present"] = 1
                    features["phone_conf"] = conf
                    phone_feats["phone_x1"] = int(x1)
                    phone_feats["phone_y1"] = int(y1)
                    phone_feats["phone_x2"] = int(x2)
                    phone_feats["phone_y2"] = int(y2)
                    phone_feats["phone_present"] = 1
                    self.phone_features = phone_feats
                    return features  #if you find that is mobile then automatically return it.
                
        return features
    
