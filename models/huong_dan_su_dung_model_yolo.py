from ultralytics import YOLO

model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\weights\yolo\best.onnx")

result = model.predict(
    source= r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\data\181829-867371711_small.mp4",
    imgsz=2650,
    show=True,
    save=True
)