from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model("test.jpg", show=True)