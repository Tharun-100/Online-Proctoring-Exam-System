from ultralytics import YOLO

# this will automatically download the pretrained weights if not present
model = YOLO("yolo26n.pt")