import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import supervision as sv
from ultralytics import YOLO
import time
import numpy as np 
from tqdm import tqdm 
MODEL_PATH = './weights/yolo/best.onnx'

VIDEO_PATH = r'C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\data\test\14443854_1920_1080_60fps.mp4'

DEVICE = 'cpu' # '0' de chuyen sang GPU

IMG_SIZE = 640         
CONF_THRESH = 0.5
FRAME_SKIP = 2         
SAVE_OUTPUT = True
OUTPUT_PATH = "outputs/yolo/detect/video_output.mp4" 


WINDOW_NAME = "🔍 YOLO ONNX + Supervision (Video)"


print(f"🔹 Đang tải model ONNX: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


print(f"🎥 Đang mở video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception(f"❌ Không thể mở file video: {VIDEO_PATH}")


w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Tính toán FPS cho video output
output_fps = fps / FRAME_SKIP
print(f"Thông tin video: {w}x{h}, {fps:.2f} FPS, Tổng số {total_frames} frames.")
print(f"Video output sẽ có {output_fps:.2f} FPS (do skip {FRAME_SKIP-1} frame).")

if SAVE_OUTPUT:
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (w, h))
    print(f"💾 Kết quả sẽ được lưu tại: {OUTPUT_PATH}")

# Cho phép thay đổi kích thước cửa sổ
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
print("🚀 Bắt đầu xử lý... Nhấn 'q' để dừng sớm.")

frame_count = 0
frame_processed = 0

#Biến để làm mượt (Smooth) thông số
SMOOTHING_WINDOW = 10
latencies_pre = []
latencies_inf = []
latencies_post = []
latencies_loop = []


# Thanh tiến trình TQDM
pbar = tqdm(total=total_frames, desc="Đang xử lý video", unit="frame")


while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Đã xử lý hết video.")
        break

    frame_count += 1
    pbar.update(1) # Cập nhật thanh tiến trình cho mỗi frame ĐỌC

    if frame_count % FRAME_SKIP != 0:
        continue 

    # Bắt đầu đo thời gian xử lý
    proc_time_start = time.time()
    
    # DỰ ĐOÁN
    results = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
    
    # Lấy thông tin độ trễ chi tiết từ `results.speed`
    speed_dict = results.speed 
    
    latencies_pre.append(speed_dict.get('preprocess', 0))
    latencies_inf.append(speed_dict.get('inference', 0))
    latencies_post.append(speed_dict.get('postprocess', 0))
    
    
    detections = sv.Detections.from_ultralytics(results)
    tracked = tracker.update_with_detections(detections)

    # VẼ LABEL & BOX
    class_names = results.names 
    
    if len(tracked) > 0:
        xyxy = tracked.xyxy
        confs = tracked.confidence
        class_ids = tracked.class_id

        labels = [
            f"{class_names[int(c)]} {conf:.2f}"
            for c, conf in zip(class_ids, confs)
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=tracked
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=tracked,
            labels=labels
        )
    else:
        annotated_frame = frame.copy()

    # Kết thúc đo thời gian xử lý
    proc_time_total_ms = (time.time() - proc_time_start) * 1000 
    latencies_loop.append(proc_time_total_ms)

    # Làm mượt giá trị (tính trung bình của N frame gần nhất)
    if len(latencies_loop) > SMOOTHING_WINDOW:
        latencies_pre.pop(0)
        latencies_inf.pop(0)
        latencies_post.pop(0)
        latencies_loop.pop(0)

    avg_pre_ms = np.mean(latencies_pre)
    avg_inf_ms = np.mean(latencies_inf)
    avg_post_ms = np.mean(latencies_post)
    avg_loop_ms = np.mean(latencies_loop)
    avg_loop_fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0
    # -----------------------------------------------------------
    
    # Cập nhật thông tin cho pbar
    pbar.set_postfix_str(f"Proc FPS: {avg_loop_fps:.1f} (Inf: {avg_inf_ms:.1f}ms)")
    frame_processed += 1

    # === HIỂN THỊ THÔNG SỐ (đã được làm mượt) ===
    cv2.putText(annotated_frame, f"Proc FPS: {avg_loop_fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"  Pre: {avg_pre_ms:.1f} ms", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"  Inf: {avg_inf_ms:.1f} ms", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"  Post: {avg_post_ms:.1f} ms", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Skipping: {FRAME_SKIP-1} frame(s)", (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

    # HIỂN THỊ & GHI
    cv2.imshow(WINDOW_NAME, annotated_frame)
    if SAVE_OUTPUT:
        out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nNgười dùng đã dừng xử lý.")
        break


pbar.close()
cap.release()
if SAVE_OUTPUT:
    out.release()
cv2.destroyAllWindows()
print(f"Hoàn tất! Đã xử lý {frame_processed} frame.")
print(f"Video kết quả đã lưu tại: {OUTPUT_PATH}")
