from ultralytics import YOLO

# Nạp model YOLOv12
model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\weights\yolo\best.pt")

# Xuất sang ONNX
model.export(
    format="onnx",       # Định dạng ONNX
    opset=12,            # phiên bản ONNX opset
    dynamic=True,        # cho phép input kích thước động
    simplify=True,       # tự tối ưu graph ONNX
    imgsz=(320, 320)     # Kích thước đầu vào (H, W)
)
