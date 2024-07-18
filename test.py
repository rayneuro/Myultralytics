from ultralytics import YOLO


# Initialize model



model = YOLO("yolov8l-obb.yaml")

model.train(data="10per.yaml", epochs=1300, imgsz=640 )


