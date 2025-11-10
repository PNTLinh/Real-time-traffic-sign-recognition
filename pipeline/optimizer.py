from ultralytics import YOLO

model = YOLO(r"local_root\weights\yolo\best.pt")

# Xuất sang ONNX
model.export(
    format="onnx",       # Định dạng ONNX
    opset=12,            # phiên bản ONNX opset
    dynamic=True,        # cho phép input kích thước động
    simplify=True,       # tự tối ưu graph ONNX
    imgsz=(320, 320)     # Kích thước đầu vào (H, W)
)
