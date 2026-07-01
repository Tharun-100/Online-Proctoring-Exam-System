
"""
Smartphone Detection using YOLOv8 (Pretrained on COCO)
======================================================
YOLOv8 is pretrained on the COCO dataset which includes the 'cell phone' class (class ID 67).
No additional training needed — it works out of the box!

Install dependencies:
    pip install ultralytics opencv-python pillow

Usage:
    python detect_smartphones.py --image path/to/image.jpg
    python detect_smartphones.py --image path/to/image.jpg --model yolov8x  # larger = more accurate
    python detect_smartphones.py --webcam  # real-time webcam detection
"""

import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# COCO class ID for cell phone
CELL_PHONE_CLASS_ID = 67
CELL_PHONE_CLASS_NAME = "cell phone"


def detect_smartphones(
    source,
    model_name="yolov9m",  # n=nano, s=small, m=medium, l=large, x=extra-large
    conf_threshold=0.25,
    save_output=True,
    show=True,
):

    """
    Detect smartphones in an image or video using pretrained YOLOv8.
    Args:
        source: Path to image/video file, or 0 for webcam
        model_name: YOLOv8 variant — trade speed for accuracy:
                    yolov8n (fastest) → yolov8s → yolov8m → yolov8l → yolov8x (most accurate)
        conf_threshold: Minimum confidence score (0-1)
        save_output: Whether to save annotated result
        show: Whether to display the result     window
    """

    print(f"\n{'='*55}")
    print(f"  Smartphone Detector — YOLOv8 (COCO pretrained)")
    print(f"{'='*55}")
    print(f"  Model    : {model_name}.pt")
    print(f"  Source   : {source}")
    print(f"  Conf     : {conf_threshold}")
    print(f"{'='*55}\n")

    # Load model (auto-downloads on first run ~6MB for nano)
    print(f"[+] Loading {model_name} model...")
    model = YOLO(f"{model_name}.pt")

    # Run inference — filter to cell phone class only
    print(f"[+] Running detection...")
    
    results = model(
        source,
        conf=conf_threshold,
        classes=[CELL_PHONE_CLASS_ID],  # Only detect cell phones
        show=show,
        save=save_output,
        line_width=2,
        show_conf=True,
        show_labels=True,
    )

    # Parse and print results
    total_phones = 0
    for i, result in enumerate(results):
        boxes = result.boxes
        phones = [b for b in boxes if int(b.cls) == CELL_PHONE_CLASS_ID]
        n = len(phones)
        total_phones += n

        print(f"\n[Frame {i+1}] Detected {n} smartphone(s):")
        for j, box in enumerate(phones):
            conf = float(box.conf)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            w, h = x2 - x1, y2 - y1
            print(f"  Phone #{j+1}: confidence={conf:.2%}  bbox=[{x1},{y1} → {x2},{y2}]  size={w}×{h}px")

    print(f"\n{'='*55}")
    print(f"  Total smartphones detected: {total_phones}")
    if save_output:
        print(f"  Saved to: runs/detect/")
    print(f"{'='*55}\n")

    return results


def detect_from_webcam(model_name="yolov8n", conf_threshold=0.5):
    """Real-time smartphone detection from webcam. Press 'q' to quit."""
    print("[+] Starting webcam... Press 'q' to quit.")
    model = YOLO(f"{model_name}.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, classes=[CELL_PHONE_CLASS_ID], verbose=False)
        annotated = results[0].plot()

        n = len(results[0].boxes)
        label = f"Smartphones detected: {n}"
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
        cv2.imshow("Smartphone Detector (YOLOv8)", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect smartphones using pretrained YOLOv8")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for real-time detection")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model size (n=fastest, x=most accurate)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated output")
    parser.add_argument("--no-show", action="store_true", help="Don't display result window")
    args = parser.parse_args()

    if args.webcam:
        detect_from_webcam(model_name=args.model, conf_threshold=args.conf)

    elif args.image:
        detect_smartphones(
            source=args.image,
            model_name=args.model,
            conf_threshold=args.conf,
            save_output=not args.no_save,
            show=not args.no_show,
        )

    elif args.video:
        detect_smartphones(
            source=args.video,
            model_name=args.model,
            conf_threshold=args.conf,
            save_output=not args.no_save,
            show=not args.no_show,
        )

    else:
        print("Usage examples:")
        print("  python app2.py --image photo.jpg")
        print("  python app2.py --image photo.jpg --model yolov8x")
        print("  python app2.py --webcam")
        print("  python app2.py --video clip.mp4 --conf 0.4")
        print("\nRun with --help for all options.")