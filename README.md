# Real-time-traffic-sign-recognition

### Mục tiêu

Xây dựng hệ thống nhận diện biển báo giao thông thời gian thực sử dụng YOLOv12 để phát hiện biển báo và Vision-Language Model (VLM) để hiểu và phân loại ngữ nghĩa của biển báo. Hệ thống có khả năng chạy real-time (≥25 FPS) trên GPU/Edge device.

### Cấu trúc hệ thống
```text
traffic-sign-system/
│
├── data/                        # Quản lý dữ liệu
│   ├── loader.py                # Load dataset, video stream
│   ├── preprocess.py            # Tiền xử lý ảnh (resize, norm) cho VLM và YOLO
│   ├── augmentation.py          # Data augmentation pipeline
│   ├── test/                    # Dữ liệu thử nghiệm thực tế
│   └── statistic/               # Script thống kê phân phối dữ liệu
│
├── models/                      # Định nghĩa và quản lý Models
│   ├── yolo_detector.py         # Module YOLO detection
│   ├── vlm_classifier.py        # Module VLM classification
│
├── pipeline/                    # Luồng xử lý chính
│   ├── realtime_system.py       # Tích hợp toàn bộ hệ thống real-time
│   ├── optimizer.py             # Các kỹ thuật tối ưu tốc độ (TensorRT, quantization)
│
├── evaluation/                  # Đánh giá hiệu năng
│   ├── metrics.py               # Tính toán mAP, Precision, Recall
│   ├── evaluate.py              # Script đánh giá tổng thể model
│   └── latency_yolo.py          # Đo độ trễ (latency) của YOLO
│
├── utils/                       # Các hàm tiện ích
│   ├── logger.py                # Ghi log và visualize kết quả
│   └── paths.py                 # Quản lý đường dẫn file
│
├── weights/                     # Weights và cấu hình model
│   ├── yolo/
│   │   ├── args.yaml
│   │   ├── best.onnx
│   │   ├── best.pt
│   │   ├── model_in4.json
│   │   └── get_metadata.py
│   └── vlm/
│       ├── config.json
│       └── prompt.txt
│
├── datasets/                    # Dữ liệu (Chia tỉ lệ 7/2/1)
│   ├── raw/                     # Dữ liệu thô
│   └── processed/               # Dữ liệu đã qua xử lý & augmentation
│
├── outputs/                     # Nơi lưu kết quả đầu ra
│   ├── logs/
│   └── yolo/
│
├── main.py                      # Entry point chạy hệ thống real-time
├── requirements.txt             # Các thư viện phụ thuộc
├── config.yaml                  # File cấu hình toàn cục
└── README.md                    # Tài liệu dự án
```
### Đặc tả dữ liệu
Bộ dữ liệu biển báo giao thông Việt Nam(vietnam-traffic-sign-vr1a7-ecrhf-xasim) có số lượng là 545 gồm 19 classes: Cấm đỗ xe, Cấm dừng đỗ xe, Cấm ngược chiều, Cấm ô tô, Cấm quay đầu, Cấm rẽ phải, Cấm rẽ trái, Dừng lại, Đường không bằng phẳng,Đường không ưu tiên, Đường ưu tiên, Người đi bộ, Tốc độ 30, Tốc độ 40, Tốc độ 50, Tốc độ 60, Tốc độ 80, Trẻ em qua đường, Vòng xuyến.

Sau khi loại đi class hiếm, sử dụng các loại augmentation như: lật ngang, dọc, xoay, dịch, phóng to, thu nhỏ, cắt ảnh, thay đổi độ sáng, độ tương phản, thay đổi kênh màu, dùng Gaussian làm nhiễu mờ với xác suất khác nhau.
Dữ liệu thu được sau xử lý thu được 2486 bản ghi:

- Hình ảnh sau xử lý: 320x320, gồm các file ảnh có định dạng ".jpg", ".jpeg", ".png".
- Nhãn: gồm các thông số <class_id> <x_center> <y_center> <width> <height>
   - <class_id>: Đây là một số nguyên đại diện cho mã lớp của đối tượng.
   - <x_center>: Tọa độ tâm theo trục X của bounding box (khung chứa đối tượng).
   - <y_center>: Tọa độ tâm theo trục Y của bounding box.
   - <width>: Chiều rộng của bounding box.
   - <height>: Chiều cao của bounding box.

### Cách thực hiện 
1. Cài đặt môi trường

1.1 Clone project
```text
git clone https://github.com/PNTLinh/Real-time-traffic-sign-recognition.git
cd Real-time-traffic-sign-recognition
```
1.2 Tạo môi trường Python

py -3.10 -m venv prodl

prodl\Scripts\activate   # Windows

1.3 Cài dependencies
pip install -r requirements.txt

📦 2. Chuẩn bị config.yaml


🏋️‍♂️ 3. Huấn luyện YOLO
python main.py --mode train \
    --data datasets/processed/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 320 \
    --model-size n


Model tốt nhất sẽ được lưu tại:

outputs/yolo/train/weights/best.pt

🎥 4. Chạy realtime (webcam / video / image)
4.1 Webcam (mặc định camera 0)
python main.py --mode inference --source webcam

4.2 Chạy video input
python main.py --mode inference --source video \
    --input data/test/traffic.mp4

4.3 Chạy 1 ảnh
python main.py --mode inference --source image \
    --input data/test/sign.jpg

🧠 5. Tắt VLM (YOLO only)
python main.py --mode inference --source webcam --no-vlm

⚡ 6. Tối ưu hóa YOLO (ONNX / TensorRT / Benchmark)
6.1 Xuất ONNX
python main.py --mode optimize --optimize-action onnx --model weights/yolo/best.pt

6.2 Xuất TensorRT
python main.py --mode optimize --optimize-action tensorrt --model weights/yolo/best.onnx

6.3 Benchmark tốc độ model
python main.py --mode optimize --optimize-action benchmark \
    --model weights/yolo/best.pt \
    --iterations 200

🧪 7. Test nhanh YOLO
python main.py --mode test --input data/test/sample.jpg


Output sẽ được lưu tại:

outputs/test_result.jpg






































