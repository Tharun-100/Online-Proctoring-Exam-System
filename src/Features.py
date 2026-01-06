import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from ultralytics import YOLO

# ===============================
# LOAD MEDIAPIPE MODELS
# ===============================

base_face_det = python.BaseOptions(
    model_asset_path=r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\mediapipe_models\blaze_face_short_range.tflite"
)

face_det_options = vision.FaceDetectorOptions(
    base_options=base_face_det,
    min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(face_det_options)

base_face_lm = python.BaseOptions(
    model_asset_path=r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\mediapipe_models\face_landmarker.task"
)

face_lm_options = vision.FaceLandmarkerOptions(
    base_options=base_face_lm,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_lm_options)

base_hand = python.BaseOptions(
    model_asset_path=r"E:\Projects in ML\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\mediapipe_models\hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=base_hand,
    num_hands=2
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ===============================
# LOAD PHONE DETECTOR (YOLO)
# ===============================
phone_model = YOLO("yolov8n.pt")  # COCO class 67 = cellphone

# ===============================
# FEATURE ORDER (STRICT)
# ===============================

FEATURE_ORDER = [
    "face_present","no_of_face","face_x","face_y","face_w","face_h",
    "left_eye_x","left_eye_y","right_eye_x","right_eye_y",
    "nose_tip_x","nose_tip_y","mouth_x","mouth_y","face_conf",
    "hand_count","left_hand_x","left_hand_y","right_hand_x","right_hand_y",
    "hand_obj_interaction",
    "head_pose","head_pitch","head_yaw","head_roll",
    "phone_present","phone_loc_x","phone_loc_y","phone_conf",
    "gaze_on_script","gaze_direction",
    "gazePoint_x","gazePoint_y",
    "pupil_left_x","pupil_left_y","pupil_right_x","pupil_right_y"
]

# ===============================
# MAIN EXTRACTION FUNCTION
# ===============================

def extract_features(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    mp_img = vision.Image(image_format=vision.ImageFormat.SRGB, data=img)
    
    feats = {k: 0 for k in FEATURE_ORDER}

    # -------- FACE DETECTION --------
    face_det = face_detector.detect(mp_img)
    feats["no_of_face"] = len(face_det.detections)
    feats["face_present"] = int(feats["no_of_face"] > 0)

    if feats["face_present"]:
        bbox = face_det.detections[0].bounding_box
        feats["face_x"] = bbox.origin_x
        feats["face_y"] = bbox.origin_y
        feats["face_w"] = bbox.width
        feats["face_h"] = bbox.height
        feats["face_conf"] = face_det.detections[0].confidence[0]

    # -------- FACE LANDMARKS --------
    face_lm = face_landmarker.detect(mp_img)
    if face_lm.face_landmarks:
        lm = face_lm.face_landmarks[0]

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        feats["left_eye_x"], feats["left_eye_y"] = pt(33)
        feats["right_eye_x"], feats["right_eye_y"] = pt(263)
        feats["nose_tip_x"], feats["nose_tip_y"] = pt(1)
        feats["mouth_x"], feats["mouth_y"] = pt(13)

        # head pose
        mat = face_lm.facial_transformation_matrixes[0]
        feats["head_pose"] = 1
        feats["head_pitch"] = mat[0][0]
        feats["head_yaw"] = mat[1][1]
        feats["head_roll"] = mat[2][2]

        feats["gaze_on_script"] = 1
        feats["gaze_direction"] = 0
        feats["gazePoint_x"], feats["gazePoint_y"] = pt(168)
        feats["pupil_left_x"], feats["pupil_left_y"] = pt(468)
        feats["pupil_right_x"], feats["pupil_right_y"] = pt(473)

    # -------- HAND LANDMARKS --------
    hand_res = hand_landmarker.detect(mp_img)
    feats["hand_count"] = len(hand_res.hand_landmarks)

    if feats["hand_count"] > 0:
        l = hand_res.hand_landmarks[0][0]
        feats["left_hand_x"] = int(l.x * w)
        feats["left_hand_y"] = int(l.y * h)

    if feats["hand_count"] > 1:
        r = hand_res.hand_landmarks[1][0]
        feats["right_hand_x"] = int(r.x * w)
        feats["right_hand_y"] = int(r.y * h)

    feats["hand_obj_interaction"] = int(feats["hand_count"] > 0)

    # -------- PHONE DETECTION --------
    yolo_res = phone_model(img, verbose=False)[0]
    phones = [b for b in yolo_res.boxes if int(b.cls) == 67]

    if phones:
        b = phones[0]
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        feats["phone_present"] = 1
        feats["phone_loc_x"] = (x1 + x2) // 2
        feats["phone_loc_y"] = (y1 + y2) // 2
        feats["phone_conf"] = float(b.conf)

    return feats


if __name__ == "__main__":
    f = extract_features("input.jpg")
    print(list(f.keys()))
    print(list(f.values()))
