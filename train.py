from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained base

model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640
)