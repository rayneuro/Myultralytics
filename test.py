from ultralytics import YOLO


# Initialize model



model = YOLO("yolov8l-obb.yaml")

model.train(data="testing.yaml", epochs=10, imgsz=640)


