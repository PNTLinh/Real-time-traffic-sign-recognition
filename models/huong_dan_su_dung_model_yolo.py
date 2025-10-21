from ultralytics import YOLO

model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\weights\yolo\best.pt")  # Sử dụng model đã được huấn luyện trước đó

results = model.predict(
    source=r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\datasets\processed\processed_test\images\1bc9a8b5e69221cc788336_jpg.rf.b6e3c3f345fb6f264dca0d34aa28a9f6.jpg", # gan source la duong dan den anh hoac video can du doan
    imgsz=320,       # kích thước resize về cùng với khi train
    conf=0.25,       # ngưỡng confidence (0.25 là mức ổn)
    save=True,       # lưu ảnh kết quả có bounding box
    show=True        # hiện ảnh ra màn hình (có thể tắt nếu chạy server)
)
# Sau do no se in ra ket qua du doan tren anh/video va luu anh/video co bounding box vao thu muc runs/predict

