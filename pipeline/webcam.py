import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import supervision as sv
from ultralytics import YOLO
import time

# ==== CẤU HÌNH ====
MODEL_PATH = r"local_root\weights\yolo\best.onnx"
IMG_SIZE = 1024
CONF_THRESH = 0.2
FRAME_SKIP = 3
SAVE_OUTPUT = True
OUTPUT_PATH = "local_root\outputs\yolo\webcam_output.mp4"
OUTPUT_FPS = 25 # Target FPS for the output video
WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720
print(" Đang tải model ONNX...")
model = YOLO(MODEL_PATH, task="detect")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# ==== MỞ WEBCAM ====
cap = cv2.VideoCapture(0) # Use 0 for default webcam
if not cap.isOpened():
    raise Exception("❌ Không thể mở webcam! Kiểm tra xem webcam có được kết nối không.")

# --- Đặt độ phân giải mong muốn ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
# -----------------------------------

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set FPS tren webcam
webcam_fps = cap.get(cv2.CAP_PROP_FPS)
if webcam_fps <= 0:
    print("⚠️ Không thể lấy FPS từ webcam, mặc định là 30.")
    webcam_fps = 30

print(f"🎥 Webcam mở thành công: {w}x{h} @ {webcam_fps:.2f} FPS")


if SAVE_OUTPUT:
    # Use OUTPUT_FPS for the writer, as this is the target saved FPS
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), OUTPUT_FPS, (w, h))
    if not out.isOpened():
        print(f"❌ Không thể tạo file video tại: {OUTPUT_PATH}")
        SAVE_OUTPUT = False # Disable saving if it fails
    else:
        print(f"💾 Kết quả sẽ được lưu tại: {OUTPUT_PATH}")

print("🚀 Bắt đầu xử lý... Nhấn 'q' trên cửa sổ video để thoát.")

# --- THÊM DÒNG NÀY ĐỂ CHO PHÉP RESIZE CỬA SỔ ---
WINDOW_NAME = "🔍 YOLO ONNX + Supervision (Webcam)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# ----------------------------------------------

frame_count = 0
frame_processed = 0
last_display_time = time.time()
processing_fps = 0

# ==== VÒNG LẶP CHÍNH ====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Không nhận được khung hình từ webcam. Kết thúc.")
            break

        frame_count += 1
        
        # Bỏ qua frame nếu FRAME_SKIP > 1
        if frame_count % FRAME_SKIP != 0:
            continue

        start_time = time.time() # Bắt đầu đếm thời gian xử lý

        # === DỰ ĐOÁN ===
        # verbose=False để tắt log của YOLO cho mỗi frame
        results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # === TRACKING ===
        # Chỉ update tracker nếu có detections
        if len(detections) > 0:
            tracked = tracker.update_with_detections(detections)
        else:
            # Nếu không có gì, gọi update rỗng để tracker có thể xử lý các track cũ
            tracked = tracker.update_with_detections(sv.Detections.empty())


        # === VẼ LABEL & BOX ===
        annotated_frame = frame.copy() # Luôn bắt đầu từ frame gốc
        
        if len(tracked) > 0:
            # Lấy thông tin từ results.names
            labels = [
                f"{model.names[int(c)]} {conf:.2f}"
                for c, conf in zip(tracked.class_id, tracked.confidence)
            ]

            # 1. Vẽ boxes
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame,
                detections=tracked
                # Xóa 'labels=labels' khỏi đây
            )
            # 2. Vẽ labels
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=tracked,
                labels=labels
            )
        
        # === TÍNH TOÁN & HIỂN THỊ FPS ===
        # Tính FPS dựa trên thời gian xử lý thực tế
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Tránh chia cho 0 nếu xử lý quá nhanh
        if processing_time > 0:
            # Tính FPS của riêng phần xử lý (model + drawing)
            # Nhân với FRAME_SKIP để ước tính FPS nếu xử lý mọi frame
            processing_fps = (1.0 / processing_time) 

        cv2.putText(annotated_frame, f"Proc FPS: {processing_fps:.1f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Skipping: {FRAME_SKIP-1} frames", (10, 80), # <-- SỬA LỖI Ở ĐÂY (từ annot2 thành annotated_frame)
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


        # === HIỂN THỊ & GHI ===
        cv2.imshow(WINDOW_NAME, annotated_frame) # Đảm bảo tên cửa sổ khớp
        
        if SAVE_OUTPUT and out.isOpened():
            out.write(annotated_frame)

        frame_processed += 1
        
        # Phím thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Người dùng nhấn 'q', đang thoát...")
            break
except KeyboardInterrupt:
    print("\n🛑 Đã phát hiện KeyboardInterrupt, đang dừng...")

# ==== DỌN DẸP ====
print(f"\n✅ Hoàn tất! Đã xử lý tổng cộng {frame_processed} frame.")
cap.release()
print("🔌 Webcam đã được giải phóng.")
if SAVE_OUTPUT and out.isOpened():
    out.release()
    print(f"💾 File video đã được lưu tại: {OUTPUT_PATH}")
cv2.destroyAllWindows()
print("Cleanup complete.")


