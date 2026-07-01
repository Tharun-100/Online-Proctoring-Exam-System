import cv2
import os
import time

# ====== CONFIG ======
BASE_DIR = "New_setup/videos_for_extraction"
CLASSES = {
    '1': "normal",
    '2': "phone",
    '3': "multiple_people",
    '4': "head_pose_away",
    '5': "no_face"
}

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20

# ====================
# Create folders if not exist
for cls in CLASSES.values():
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

current_class = "normal"
recording = False
out = None

print("\n🎥 Controls:")
print("1: normal | 2: phone | 3: multiple_people | 4: head_pose_away | 5: no_face")
print("R: Start/Stop Recording | Q: Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # Overlay info
    cv2.putText(display_frame, f"Class: {current_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status = "Recording..." if recording else "Idle"
    color = (0, 0, 255) if recording else (255, 255, 255)

    cv2.putText(display_frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow("Dataset Recorder", display_frame)

    if recording and out is not None:
        out.write(frame)

    key = cv2.waitKey(1) & 0xFF

    # Change class
    if chr(key) in CLASSES:
        current_class = CLASSES[chr(key)]
        print(f"Switched to: {current_class}")
        
    # Start/Stop recording
    elif key == ord('r'):
        recording = not recording

        if recording:
            timestamp = int(time.time())
            filename = f"{current_class}_{timestamp}.mp4"
            filepath = os.path.join(BASE_DIR, current_class, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            print(f"▶ Recording started: {filepath}")
        else:
            if out:
                out.release()
                out = None
            print("⏹ Recording stopped and saved.")

    # Quit
    elif key == ord('q'):
        break

cap.release()
if out:
    out.release()

cv2.destroyAllWindows()
