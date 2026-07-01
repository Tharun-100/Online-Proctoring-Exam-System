import os
import cv2
import pandas as pd
import csv
from feature_extraction import FeatureExtractor

# -------------------------------
# CONFIG
# -------------------------------

DATASET_PATH = r"C:\Users\Tharun\Desktop\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\videos"  # Adjust to your dataset path
OUTPUT_CSV = "features.csv"
WRITE_EXCEL = False  # Set True if you want to also create Excel after CSV
FRAME_SKIP = 4   # sample every 5th frame

# -------------------------------
# INIT FEATURE EXTRACTOR
# -------------------------------
extractor = FeatureExtractor()

# -------------------------------
# COLUMN NAMES (as you defined)
# -------------------------------

columns = [
    "face_present","no_of_face","face_x","face_y","face_w","face_h",
    "left_eye_x","left_eye_y","right_eye_x","right_eye_y",
    "nose_tip_x","nose_tip_y","mouth_x","mouth_y","face_conf",
    "hand_count","left_hand_x","left_hand_y","right_hand_x","right_hand_y",
    "hand_obj_interaction","head_pose","head_pitch","head_yaw","head_roll",
    "phone_present","phone_loc_x","phone_loc_y","phone_conf",
    "gaze_on_script","gaze_direction","gazePoint_x","gazePoint_y",
    "pupil_left_x","pupil_left_y","pupil_right_x","pupil_right_y","label"
]

# -------------------------------
# MAIN PIPELINE
# -------------------------------

data = [] if WRITE_EXCEL else None
rows_written = 0

csv_path = os.path.abspath(OUTPUT_CSV)
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(columns)

for label_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_folder)

    if not os.path.isdir(folder_path):
        continue

    # Label logic (adjust if needed)
    label = 1 if "suspicious" in label_folder.lower() else 0

    for video_file in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_file)

        print(f"Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_SKIP == 0:
                try:
                    features = extractor.extract_features(frame)

                    # Build a row in the exact column order (excluding label)
                    row = [features.get(col, 0) for col in columns[:-1]]
                    if len(row) != len(columns) - 1:
                        print(f"Feature size mismatch: {len(row)}")
                        continue

                    row.append(label)
                    if WRITE_EXCEL:
                        data.append(row)
                    csv_writer.writerow(row)
                    rows_written += 1

                except Exception as e:
                    print(f"Error processing frame: {e}")

            frame_count += 1

        cap.release()

csv_file.close()

print(f"\n Done! CSV saved to: {csv_path} (rows: {rows_written})")

if WRITE_EXCEL:
    df = pd.DataFrame(data, columns=columns)
    output_xlsx = os.path.abspath("features.xlsx")
    df.to_excel(output_xlsx, index=False)
    print(f"Excel saved to: {output_xlsx}")
