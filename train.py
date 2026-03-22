from ultralytics import YOLO

model = YOLO("yolov8s.pt")  

model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640
)