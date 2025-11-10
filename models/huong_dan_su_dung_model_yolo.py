from ultralytics import YOLO

model = YOLO(r"local_root\weights\yolo\best.onnx")

result = model.predict(
    source= r"local_root\data\test\14443854_1920_1080_60fps.mp4",
    imgsz=2560,
    show=True,
    save=True
)


