import cv2
import os
import time

# Create dataset folders
base_path = r"C:\Users\Tharun\Desktop\FRAUD DETECTION SYSTEM FOR THE ONLINE PROCTORED EXAMS\videos"
os.makedirs(os.path.join(base_path, "0"), exist_ok=True)
os.makedirs(os.path.join(base_path, "1"), exist_ok=True)

def get_next_filename(label):
    folder = os.path.join(base_path, label)
    count = len(os.listdir(folder))
    return os.path.join(folder, f"clip_{count+1:03d}.mp4")

def record_clip(duration=10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    filename = "temp.mp4"
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))

    start_time = time.time()
    print("Recording...")

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow("Recording (Press Q to cancel)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return filename


while True:
    input("Press ENTER to start recording...")

    video_file = record_clip(10)

    if video_file is None:
        continue

    label = input("Enter label (0 = normal, 1 = suspicious): ").strip()

    if label not in ["0", "1"]:
        print("Invalid label. Skipping...")
        continue

    save_path = get_next_filename(label)
    os.rename(video_file, save_path)

    print(f"Saved: {save_path}")

    cont = input("Record next? (y/n): ").strip().lower()
    if cont != 'y':
        break

print("Dataset collection finished.")