# src/create_dataset.py

import cv2
import os
import pandas as pd
from pathlib import Path
from feature_extraction2 import FeatureExtractor
import time
import traceback


VIDEO_DIR = r"C:\Users\Tharun\Desktop\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\New_setup\videos_for_extraction"
OUTPUT_DIR = r"C:\Users\Tharun\Desktop\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\New_setup\new_data"

FRAME_CSV = os.path.join(OUTPUT_DIR, "frame_level_features.csv")
WINDOW_CSV = os.path.join(OUTPUT_DIR, "final_window_dataset.csv")
PROGRESS_CSV = os.path.join(OUTPUT_DIR, "processed_videos.csv")

WINDOW_SECONDS = 1
SUSPICIOUS_THRESHOLD = 0.8
FRAME_BATCH_SIZE = 500
PROGRESS_EVERY_N_FRAMES = 100
RESUME = True

def write_rows_to_csv(rows, csv_path, first_write):
    if not rows:
        return first_write
    
    df = pd.DataFrame(rows)

    df.to_csv(
        csv_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False
    )

    return False


def create_window_row(window_rows, video_id, start_time, end_time):

    window_df = pd.DataFrame(window_rows)
    phone_ratio = window_df["phone_present"].mean()
    multiple_faces_ratio = window_df["multiple_faces"].mean()
    head_pose_away_ratio = window_df["head_pose_away"].mean()
    no_face = window_df["no_face"]
    no_face_ratio = float(no_face.mean()) if getattr(no_face, "size", 0) else 0.0

    label = int(
        phone_ratio >= 0.35 or
        multiple_faces_ratio >= 0.5 or
        head_pose_away_ratio >= SUSPICIOUS_THRESHOLD or
        no_face_ratio >= SUSPICIOUS_THRESHOLD
    )

    return {
        "video_id": video_id,
        "window_start_sec": start_time,
        "window_end_sec": end_time,
        "phone_present_ratio": phone_ratio,
        "max_phone_conf": window_df["phone_conf"].max(),
        "avg_num_faces": window_df["num_faces"].mean(),
        "no_face_ratio": no_face_ratio,
        "multiple_faces_ratio": multiple_faces_ratio,
        "head_pose_away_ratio": head_pose_away_ratio,
        "avg_head_pitch": window_df["head_pitch"].mean(),
        "avg_head_yaw": window_df["head_yaw"].mean(),
        "avg_head_roll": window_df["head_roll"].mean(),
        "label": label,
        "source_folder": window_df["source_folder"].iloc[0],
    }

def extract_frames_and_create_windows():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    error_log_path = os.path.join(OUTPUT_DIR, "data_video_creation_errors.log")
    processed_ids = set()

    if not RESUME:
        for path in (FRAME_CSV, WINDOW_CSV, PROGRESS_CSV, error_log_path):
            if os.path.exists(path):
                os.remove(path)
    else:
        if not os.path.exists(PROGRESS_CSV) and (os.path.exists(FRAME_CSV) or os.path.exists(WINDOW_CSV)):
            print(
                "WARNING: RESUME=True but processed_videos.csv is missing while output CSVs exist. "
                "Starting fresh to avoid duplicate rows."
            )
            for path in (FRAME_CSV, WINDOW_CSV, PROGRESS_CSV, error_log_path):
                if os.path.exists(path):
                    os.remove(path)

        if os.path.exists(PROGRESS_CSV):
            try:
                processed_df = pd.read_csv(PROGRESS_CSV)
                if "video_id" in processed_df.columns:
                    processed_ids = set(processed_df["video_id"].astype(str).tolist())
            except Exception:
                processed_ids = set()

    extractor = FeatureExtractor()

    video_paths = list(Path(VIDEO_DIR).glob("*/*.mp4"))

    if len(video_paths) == 0:
        print("No videos found. Check your VIDEO_DIR path and folder structure.")
        return
    
    first_frame_write = not (os.path.exists(FRAME_CSV) and os.path.getsize(FRAME_CSV) > 0)
    first_window_write = not (os.path.exists(WINDOW_CSV) and os.path.getsize(WINDOW_CSV) > 0)
    
    for idx, video_path in enumerate(video_paths, start=1):
        video_path = str(video_path)
        video_label_name = Path(video_path).parent.name
        video_id = Path(video_path).stem

        if video_id in processed_ids:
            print(f"\n[{idx}/{len(video_paths)}] Skipping already processed: {video_id}")
            continue

        print(f"\n[{idx}/{len(video_paths)}] Processing video: {video_path}")
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps is None:
                fps = 30

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            frame_id = 0
            frame_batch_rows = []

            current_window_start = 0
            current_window_end = WINDOW_SECONDS
            current_window_rows = []
            last_progress_t = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp_sec = frame_id / fps

                try:
                    features = extractor.extract_features(frame)
                except Exception as e:
                    print(f"Error at frame {frame_id}: {e}")
                    frame_id += 1
                    continue

                num_faces = int(features.get("number_of_people", 0))

                row = {
                    "video_id": video_id,
                    "video_path": video_path,
                    "source_folder": video_label_name,
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp_sec,

                    "phone_present": int(features.get("phone_present", 0)),
                    "phone_conf": float(features.get("phone_conf", 0.0)),
                    "no_face": int(num_faces==0),

                    "num_faces": num_faces,
                    "multiple_faces": int(num_faces > 1),

                    "head_pose_away": int(features.get("head_pose", 0)),

                    "head_pitch": float(features.get("head_pitch", 0.0)),
                    "head_yaw": float(features.get("head_yaw", 0.0)),
                    "head_roll": float(features.get("head_roll", 0.0)),
                }

                frame_batch_rows.append(row)

                if len(frame_batch_rows) >= FRAME_BATCH_SIZE:
                    first_frame_write = write_rows_to_csv(
                        frame_batch_rows,
                        FRAME_CSV,
                        first_frame_write
                    )
                    frame_batch_rows = []

                while timestamp_sec >= current_window_end:
                    if len(current_window_rows) > 0:
                        window_row = create_window_row(
                            current_window_rows,
                            video_id,
                            current_window_start,
                            current_window_end
                        )
                        first_window_write = write_rows_to_csv(
                            [window_row],
                            WINDOW_CSV,
                            first_window_write
                        )

                    current_window_rows = []
                    current_window_start = current_window_end
                    current_window_end += WINDOW_SECONDS

                current_window_rows.append(row)
                frame_id += 1

                if frame_id % PROGRESS_EVERY_N_FRAMES == 0:
                    now = time.time()
                    if now - last_progress_t >= 1.0:
                        if total_frames > 0:
                            print(f"  progress: {frame_id}/{total_frames} frames ({(frame_id/total_frames)*100:.1f}%)")
                        else:
                            print(f"  progress: {frame_id} frames")
                        last_progress_t = now

            if frame_batch_rows:
                first_frame_write = write_rows_to_csv(
                    frame_batch_rows,
                    FRAME_CSV,
                    first_frame_write
                )

            if len(current_window_rows) > 0:
                window_row = create_window_row(
                    current_window_rows,
                    video_id,
                    current_window_start,
                    current_window_end
                )
                first_window_write = write_rows_to_csv(
                    [window_row],
                    WINDOW_CSV,
                    first_window_write
                )

            print(f"Finished video: {video_id}")
            print(f"Total frames processed: {frame_id}")

            pd.DataFrame([{"video_id": video_id, "video_path": video_path}]).to_csv(
                PROGRESS_CSV,
                mode="a" if os.path.exists(PROGRESS_CSV) else "w",
                header=not (os.path.exists(PROGRESS_CSV) and os.path.getsize(PROGRESS_CSV) > 0),
                index=False,
            )

        except Exception:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== Failed video: {video_path} ===\n")
                f.write(traceback.format_exc())
            print(f"ERROR: Video failed, logged to: {error_log_path}")
            continue
        finally:
            if cap is not None:
                cap.release()

    print("\nDone.")
    print(f"Frame-level CSV saved at: {FRAME_CSV}")
    print(f"Window-level CSV saved at: {WINDOW_CSV}")


if __name__ == "__main__":
    extract_frames_and_create_windows()
