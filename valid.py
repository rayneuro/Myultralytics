from ultralytics import YOLO





valid_model = YOLO('/home/ray/Myultralytics/runs/obb/FCN_to_incomplete_10per/weights/best.pt')

valid_model.val(data='./AllData.yaml', imgsz=640,save_txt=True)
