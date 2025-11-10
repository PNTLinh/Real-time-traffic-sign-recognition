from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Nạp model YOLO
model = YOLO("yolo12n.pt")  

# Cấu hình augmentation 
augmentation = {
    "degrees": 10,             # Xoay ±10°
    "scale": 0.1,              # Zoom (90%-110%)
    "hsv_h": 0.015,            # Hue ±15°
    "hsv_s": 0.15,             # Saturation ±15%
    "hsv_v": 0.15,             # Brightness ±15%
    "flipud": 0.0,             # Không lật dọc
    "fliplr": 0.5,             # 50% lật ngang
    "mosaic": 1.0,             # bật mosaic
    "mixup": 0.1               # bật mixup nhẹ (tùy chọn)
}

model.train(
    data=r"local_root\datasets\data.yaml",   
    epochs=100,
    imgsz=320,                  # kích thước ảnh đầu vào
    batch=16,
    name="traffic_sign_v3_aug",
    workers=4,
    project="runs/train",
    exist_ok=True,
    device="0",                # sử dụng GPU đầu tiên
    **augmentation               # truyền cấu hình augmentation vào
)
