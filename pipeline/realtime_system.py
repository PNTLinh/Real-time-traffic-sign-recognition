"""
realtime_system.py
Hệ thống real-time nhận diện biển báo giao thông:
- Hỗ trợ: Image, Video, Webcam
- YOLO detection
- VLM batch classification
- Bayesian Fusion
"""

from __future__ import annotations
import cv2
import time
import threading
import os
import sys
from queue import Queue

# Chỉ import cv2_imshow nếu chạy trên Colab
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import numpy as np
from models.yolo_detector import YOLODetector
from models.vlm_classifier import VLMClassifier
from pipeline.optimizer import ModelOptimizer


# --------------------------------------------------------------
# FUSION (YOLO + VLM)
# --------------------------------------------------------------

def fusion_bayesian(p_yolo: float, p_vlm: float):
    """
    Công thức chuẩn xác hơn cộng weighted:
    p = (py * pv) / ((py*pv) + ((1-py)*(1-pv)))
    """
    return (p_yolo * p_vlm) / ((p_yolo * p_vlm) + (1 - p_yolo) * (1 - p_vlm) + 1e-8)


# --------------------------------------------------------------
# REAL-TIME SYSTEM
# --------------------------------------------------------------

class RealTimeSystem:
    def __init__(
        self,
        camera_id=0,
        img_size=320,
        show_vlm=True,
        batch_vlm=True,
    ):
        self.img_size = img_size
        self.show_vlm = show_vlm
        self.batch_vlm = batch_vlm

        # Queue for camera frames
        self.frame_queue = Queue(maxsize=3)

        # Load models
        print("🚀 Initializing models...")
        self.detector = YOLODetector("weights/yolo/best.pt")
        # Kiểm tra xem file cache có tồn tại không để tránh lỗi
        vlm_cache = "weights/vlm/text_features.pt"
        if not os.path.exists(vlm_cache):
             print(f"⚠️ Warning: VLM cache not found at {vlm_cache}. It will be created.")
        
        self.vlm = VLMClassifier(cache_path=vlm_cache)

        # Optimizer
        self.opt = ModelOptimizer()
        self.opt.warmup_yolo(self.detector.model, img_size=img_size)

        self.camera_id = camera_id
        self.cap = None
        self.running = False

    # --------------------------------------------------------------
    # CORE PROCESSING (Dùng chung cho cả 3 mode)
    # --------------------------------------------------------------
    def process_frame(self, frame):
        """
        Nhận vào 1 frame -> Detect -> Classify -> Render -> Trả về frame kết quả
        """
        # 1. YOLO detect
        dets = self.detector.detect(frame)

        # 2. Collect crops for VLM
        crops_boxes = [d["bbox"] for d in dets]

        # 3. Run VLM batch
        if self.show_vlm and len(crops_boxes) > 0:
            vlm_results = self.vlm.classify_crops(frame, crops_boxes)
        else:
            vlm_results = []

        # 4. Fusion + Drawing
        out_frame = self.render(frame, dets, vlm_results)
        return out_frame

    # --------------------------------------------------------------
    # MODES: IMAGE & VIDEO
    # --------------------------------------------------------------

    def run_image(self, image_path):
        """Chế độ xử lý ảnh tĩnh"""
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            return

        print(f"🖼 Processing Image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print("❌ Error: Could not read image.")
            return

        # Xử lý
        start_t = time.time()
        result_frame = self.process_frame(frame)
        end_t = time.time()

        print(f"✅ Processed in {end_t - start_t:.4f}s")
        
        # Hiển thị
        if IN_COLAB:
            print("🔻 Result:")
            cv2_imshow(result_frame)
        
        # Lưu file
        out_path = "output_" + os.path.basename(image_path)
        cv2.imwrite(out_path, result_frame)
        print(f"💾 Saved to: {out_path}")

    def run_video(self, video_path):
        """Chế độ xử lý video file"""
        if not os.path.exists(video_path):
            print(f"❌ Error: Video not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video.")
            return

        # Thông số video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Writer
        out_name = "output_" + os.path.basename(video_path)
        # MP4V codec tương thích tốt
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

        print(f"📹 Processing Video: {video_path} ({width}x{height} @ {fps}fps)")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Xử lý frame
            processed_frame = self.process_frame(frame)
            
            # Ghi vào file output
            out.write(processed_frame)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"⏳ Processing frame {frame_idx}/{total_frames}...", end='\r')

        cap.release()
        out.release()
        print(f"\n✅ Video processing complete! Saved to {out_name}")

    # --------------------------------------------------------------
    # MODE: LIVE WEBCAM (Giữ nguyên logic cũ)
    # --------------------------------------------------------------

    def camera_thread(self):
        print("🎥 Camera thread started")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def run(self):
        """Chế độ Webcam Live (Dùng cho local máy tính, không dùng cho Colab)"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.running = True
        
        cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()

        print("🔥 Real-time system started (Webcam Mode)")

        while True:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            
            # Tái sử dụng hàm process_frame
            out_frame = self.process_frame(frame)

            cv2.imshow("Traffic Sign System", out_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

    # --------------------------------------------------------------
    # RENDER
    # --------------------------------------------------------------

    def render(self, frame, dets, vlm_results):
        img = frame.copy()

        # Tạo map kết quả VLM theo index nếu cần, 
        # nhưng ở đây ta giả định thứ tự crop = thứ tự result
        # Nếu số lượng không khớp (do lỗi nào đó), dùng min length
        limit = min(len(dets), len(vlm_results)) if vlm_results else 0

        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            yolo_conf = det["confidence"]

            # Màu xanh lá
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {yolo_conf:.2f}"

            # Nếu có kết quả VLM cho box này
            if i < limit:
                vlm = vlm_results[i]
                vlm_label = vlm["pred_label"]
                vlm_conf = vlm["pred_score"]

                # Áp dụng Fusion Bayesian
                combined = fusion_bayesian(yolo_conf, vlm_conf)
                
                # Hiển thị thêm thông tin
                # label += f" | {vlm_label} {vlm_conf:.2f}"
                label = f"{vlm_label} | Fused: {combined:.2f}"
                
                # Highlight màu đỏ nếu độ tin cậy thấp
                if combined < 0.4:
                    color = (0, 0, 255) # Đỏ
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Vẽ nền chữ
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            
            # Vẽ chữ
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return img