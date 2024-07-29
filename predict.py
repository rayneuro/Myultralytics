from ultralytics import YOLO


model = YOLO('/home/ray/Myultralytics/runs/obb/10%+CA/weights/best.pt')

model.predict('/home/ray/Myultralytics/test_images/train' ,save = True ,save_txt=True)