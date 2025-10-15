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













































