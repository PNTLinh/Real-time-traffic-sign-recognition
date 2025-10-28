# Real-time-traffic-sign-recognition

### Mục tiêu

Xây dựng hệ thống nhận diện biển báo giao thông thời gian thực sử dụng YOLOv12 để phát hiện biển báo và Vision-Language Model (VLM) để hiểu và phân loại ngữ nghĩa của biển báo. Hệ thống có khả năng chạy real-time (≥25 FPS) trên GPU/Edge device.

### Cấu trúc hệ thống

traffic-sign-system/
│
├── data/                          # Quản lý dữ liệu
│   ├── loader.py                 # Load dataset, video
│   ├── preprocess.py             # Tiền xử lý ảnh cho cả vlm và yolo
│   └── augmentation.py           # Data augmentation
│
├── models/                        # Models
│   ├── yolo_detector.py          # YOLO detection
│   ├── vlm_classifier.py         # VLM classification
│   └── trainer.py                # Training cả 2 models
│
├── pipeline/                      # Pipeline xử lý
│   ├── realtime_system.py        # Xử lý real-time
│   └── optimizer.py              # Tối ưu tốc độ
│
├── evaluation/                    # Đánh giá
│   ├── metrics.py                # Tính mAP, Precision, Recall
│   ├── evaluate.py               # Đánh giá models
│   └── benchmark.py              # Đo FPS, latency
│
├── utils/                         # Tiện ích
│   ├── logger.py                   # Vẽ kết quả
│   
│   
│
├── configs/                       # Cấu hình
│   └── config.yaml               # Config chung
│
├── weights/                       # Model weights
│   ├── yolo/
│   └── vlm/
│
├── datasets/                      # Dữ liệu 7/2/1
│   ├── train/
│   ├── val/
│   └── test/
│
├── outputs/                       # Kết quả
│   ├── logs/
│
├── main.py                        # Chạy real-time
├── train.py                       # Training
├── evaluate.py                    # Đánh giá
├── requirements.txt
└── README.md

### Đặc tả dữ liệu
Bộ dữ liệu biển báo giao thông Việt Nam(vietnam-traffic-sign-vr1a7-ecrhf-xasim) có số lượng là 545 gồm 19 classes: 
Cấm đỗ xe, Cấm dừng đỗ xe, Cấm ngược chiều, Cấm ô tô, Cấm quay đầu, Cấm rẽ phải, Cấm rẽ trái, Dừng lại, Đường không bằng phẳng,Đường không ưu tiên, Đường ưu tiên, Người đi bộ, Tốc độ 30, Tốc độ 40, Tốc độ 50, Tốc độ 60, Tốc độ 80, Trẻ em qua đường, Vòng xuyến.

Sau khi loại đi class hiếm, sử dụng các loại augmentation như: lật ngang, dọc, xoay, dịch, phóng to, thu nhỏ, cắt ảnh, thay đổi độ sáng, độ tương phản, thay đổi kênh màu, dùng Gaussian làm nhiễu mờ với xác suất khác nhau.
Dữ liệu thu được sau xử lý thu được 2486 bản ghi:

- Hình ảnh sau xử lý: 320x320, gồm các file ảnh có định dạng ".jpg", ".jpeg", ".png".
- Nhãn: gồm các thông số <class_id> <x_center> <y_center> <width> <height>
   - <class_id>: Đây là một số nguyên đại diện cho mã lớp của đối tượng.
   - <x_center>: Tọa độ tâm theo trục X của bounding box (khung chứa đối tượng).
   - <y_center>: Tọa độ tâm theo trục Y của bounding box.
   - <width>: Chiều rộng của bounding box.
   - <height>: Chiều cao của bounding box.



### Hướng dẫn chạy mô hình yolo
- Thử nghiệm trên ảnh/video: models/huong_dan_su_dung_model_yolo.py
- Thử nghiệm trên webcam: pipeline/webcam.py









































