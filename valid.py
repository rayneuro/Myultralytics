from ultralytics import YOLO


model = YOLO('/home/ray/Myultralytics/runs/obb/10%+CA/weights/best.pt')


model.val(data = './10per.yaml', imgsz=640,save_txt=True)
