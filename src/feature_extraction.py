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

model_dir=r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\src\models"
class FeatureExtractor:
    def __init__(self, model_dir: str = r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\src\models"):
        """
        Initialize MediaPipe Tasks for feature extraction
        (Face detection, face mesh, hands, pose)

        model_dir:
            Directory containing .task models
        """

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
        self.phone_model = YOLO(os.path.join(model_dir, "yolov8n-mobile-phone.pt"))  # Mobile Detection Model for Phones
        self.phone_features={}

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

        # Extract hand features
        hand_features = self._extract_hand_features(rgb_image, width, height,features)
        features.update(hand_features)

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
            features['face_x'] = float(bbox.origin_x)
            features['face_y'] = float(bbox.origin_y)
            features['face_w'] = float(bbox.width)
            features['face_h'] = float(bbox.height)
            features['face_conf'] = detection.categories[0].score * 100
            #-------------- Face mesh landmarks--------------

            mesh_results = self.face_mesh.detect(mp_image)

            if mesh_results.face_landmarks:
                landmarks = mesh_results.face_landmarks[0]
                print("The Landmarks length is:", len(landmarks))

                # Left eye (MediaPipe landmarks: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
                left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246] # use only key points for center calculation
                left_eye_center = self._get_landmark_center(landmarks, left_eye_landmarks, width, height)
                features['left_eye_x'] = float(left_eye_center[0])
                features['left_eye_y'] = float(left_eye_center[1])

                # Right eye (MediaPipe landmarks: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
                right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

                # use only key points for center calculation
                right_eye_center = self._get_landmark_center(landmarks, right_eye_landmarks, width, height)

                features['right_eye_x'] = float(right_eye_center[0])
                features['right_eye_y'] = float(right_eye_center[1])

                # Nose tip (landmark 4)

                nose_tip = landmarks[4]
                features['nose_tip_x'] = float(nose_tip.x * width)
                features['nose_tip_y'] = float(nose_tip.y * height)

                # Mouth center (landmarks: 13, 14, 308, 324, 318)
                # mouth_landmarks = [13, 14, 308, 324, 318] # midnt of lower and upper lip: okay it is fine it seems:--
                mouth_landmarks = [61, 185, 40, 39, 37, 0, 267, 269, 270, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146] # outer lip landmarks for better center calculation
                mouth_center = self._get_landmark_center(landmarks, mouth_landmarks, width, height)
                features['mouth_x'] = float(mouth_center[0])
                features['mouth_y'] = float(mouth_center[1])
        else:
            features['face_present'] = 0
            features['no_of_face'] = 0
        return features
    
    def _extract_hand_features(self, rgb_image: np.ndarray, width: int, height: int, all_features: Dict) -> Dict:
        """Extract hand-related features"""
        features = {}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        hand_results = self.hands.detect(mp_image)
        phone_present = 0
        phone_diag = 0.0
        phone_cx = phone_cy = 0.0
        min_dist= float('inf')
        # Extract the phone features

        if self.phone_features.get('phone_present', 0) == 1:
            phone_present = 1
            x1 = self.phone_features['phone_x1']
            y1 = self.phone_features['phone_y1']
            x2 = self.phone_features['phone_x2']
            y2 = self.phone_features['phone_y2']
            phone_x=float(x1)
            phone_y=float(y1)
            phone_w=float(x2-x1)
            phone_h=float(y2-y1)
            # Phone Center:-
            phone_cx=(phone_x + phone_w)/2
            phone_cy=(phone_y + phone_h)/2
            phone_diag = math.sqrt(phone_w**2 + phone_h**2)
            

        if hand_results.hand_landmarks:
            features['hand_count'] = len(hand_results.hand_landmarks)
            # Get hand landmarks
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                # Get hand type (left or right)
                hand_type = hand_results.handedness[idx][0].category_name
                wrist = hand_landmarks[0]
            # Don't Multiply Them With The Coordinate System
                wrist_x = wrist.x #* width
                wrist_y = wrist.y #* height
                print("Wrist coordinates before scaling:", wrist_x, wrist_y)
                if hand_type == 'Left':
                    features['left_hand_x'] = wrist_x 
                    features['left_hand_y'] = wrist_y

                elif hand_type == 'Right':
                    features['right_hand_x'] = wrist_x
                    features['right_hand_y'] = wrist_y
                if phone_present == 1 and wrist_x > 0 and wrist_y > 0:
                    # Calculate distance from wrist to phone center
                    
                    # print("Wrist and phone coordinates (wx,wy,px,py):", wrist_x, wrist_y, phone_cx, phone_cy)
                    wrist_x = wrist_x * width
                    wrist_y = wrist_y * height

                    dist = math.sqrt((wrist_x - phone_cx)**2 + (wrist_y - phone_cy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
            if(self.phone_features.get('phone_present', 0) == 1 and min_dist < phone_diag ): # If hand within the phone diagonal, consider it an interaction
                features['hand_obj_interaction'] = 1
        else:
            features['hand_count'] = 0
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
        print("Head Pose Radians - Pitch:", pitch, "Yaw:", yaw, "Roll:", roll)

        # print(features["head_pitch"])
        
        # print(features["head_yaw"])

        # print(features["head_roll"])

    # -----------------------------In General pitch has more dominance over pitch:--------------------

        if abs(yaw) <= 20 and abs(pitch) <= 15:
            features["head_pose"] = "Forward"
        # elif pitch < -15:
        #     features["head_pose"] = "Up" # This is a bit tricky because if the person is looking up, the pitch will be negative, but we can set a threshold to avoid misclassification due to slight head movements. We can say that if the pitch is less than -15 degrees, we consider it as looking up.
        elif pitch > 15:
            features["head_pose"] = "Down"
        elif yaw < -15:
            features["head_pose"] = "Left"
        elif yaw > 20:
            features["head_pose"] = "Right"

        # features["nose_2d"] = (int(landmarks[1].x * width),int(landmarks[1].y * height))
        # features["nose_3d"]=(int(landmarks[1].x * width), int(landmarks[1].y * height),int( landmarks[1].z * 3000))
        # features["rvec"] = rvec
        # features["tvec"] = tvec
        # features["camera_matrix"] = camera_matrix
        # features["dist_coeffs"] = dist_coeffs     
        return features

    def _extract_gaze_features(self, rgb_image: np.ndarray, width: int, height: int) -> Dict:
        """Extract gaze-related features"""
        features = {'gaze_on_script': 0,'gaze_direction': 'None','gazePoint_x': 0,'gazePoint_y': 0,'pupil_left_x': 0,'pupil_left_y': 0,'pupil_right_x': 0,'pupil_right_y': 0} #,'dx':0,'dy':0}
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_image)
        result = self.face_mesh.detect(mp_image)

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
        # 1. Pupil positions

        features['pupil_left_x'] = left_iris_x
        features['pupil_left_y'] = left_iris_y
        features['pupil_right_x'] = right_iris_x
        features['pupil_right_y'] = right_iris_y
        # 2. Eye center (average pupil)

        gaze_point_x = (left_iris_x + right_iris_x) / 2
        gaze_point_y = (left_iris_y + right_iris_y) / 2
        features['gazePoint_x'] = int(gaze_point_x)
        features['gazePoint_y'] = int(gaze_point_y)

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
        # # Storing dx and dy for better understanding of the gaze direction in the future:
        # features['dx'] = dx
        # features['dy'] = dy
        
        CENTER_TH_X = 0.08 * width    # ~8% of width
        CENTER_TH_Y = 0.08 * height   # ~8% of height
        DIR_TH_X = 0.12 * width       # directional threshold
        DIR_TH_Y = 0.12 * height

        # Use only the target label set expected by the downstream model:
        # Center, Top_left, Top_right, Bottom_left, Bottom_right, None
        if abs(dx) < CENTER_TH_X and abs(dy) < CENTER_TH_Y:
            gaze_direction = 'center'
        elif abs(dx) < DIR_TH_X or abs(dy) < DIR_TH_Y:
            # Ambiguous direction (not centered, but not strongly diagonal)
            gaze_direction = 'None'
        elif dx >= 0 and dy >= 0:
            gaze_direction = 'top_right'
        elif dx < 0 and dy >= 0:
            gaze_direction = 'top_left'
        elif dx >= 0 and dy < 0:
            gaze_direction = 'bottom_right'
        else:
            gaze_direction = 'bottom_left'
            
        features['gaze_direction'] = gaze_direction

        # Gaze on script determination using simple heuristics

          # Need to set a proper thresholds for this:-
        
        if gaze_direction in ['bottom_right', 'bottom_left'] or gaze_point_y > height*0.6:
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
        phone_feats={}
        #  self model :--->
        results = self.phone_model(image, conf=0.5)  # Adjust confidence threshold as needed
        
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                # cls = int(b.cls[0])
                conf= float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0].tolist()  # Get bounding box coordinates
                features['phone_present'] = 1
                phone_feats['phone_x1'] = int(x1)
                phone_feats['phone_y1'] = int(y1)
                phone_feats['phone_x2'] = int(x2)
                phone_feats['phone_y2'] = int(y2)
                phone_feats['phone_present'] = 1
                features['phone_loc_x'] = int((x1 + x2) / 2)
                features['phone_loc_y'] = int((y1 + y2) / 2)
                features['phone_conf'] = conf
                self.phone_features = phone_feats
                return features

        # results = self.phone_model(image)
        # for result in results:
        #     if result.boxes is None:
        #         continue
        #     for box in result.boxes:
        #         cls = int(box.cls[0])
        #         if cls == 67:
        #             features['phone_present'] = 1
        #             x1, y1, x2, y2 = box.xyxy[0]
        #             phone_feats['phone_x1'] = int(x1)
        #             phone_feats['phone_y1'] = int(y1)
        #             phone_feats['phone_x2'] = int(x2)
        #             phone_feats['phone_y2'] = int(y2)
        #             phone_feats['phone_present'] = 1
        #             features['phone_loc_x'] = int((x1 + x2) / 2)
        #             features['phone_loc_y'] = int((y1 + y2) / 2)
        #             features['phone_conf'] = float(box.conf[0])
        #             self.phone_features = phone_feats
        #             return features # Return on first detected phone
        return features
    
    def _get_landmark_center(self, landmarks, landmark_indices: list, width: int, height: int) -> Tuple[float, float]:
        """Calculate center point of specified landmarks"""
        x_coords = [landmarks[i].x * width for i in landmark_indices]
        y_coords = [landmarks[i].y * height for i in landmark_indices]
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
