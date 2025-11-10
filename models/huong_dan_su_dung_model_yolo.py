from ultralytics import YOLO

model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\weights\yolo\best.onnx")

result = model.predict(
    source= r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\14443854_1920_1080_60fps.mp4",
    imgsz=2560,
    show=True,
    save=True
)


