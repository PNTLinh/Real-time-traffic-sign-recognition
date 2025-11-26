"""
realtime_system.py
Hệ thống real-time nhận diện biển báo giao thông:
- Camera Thread (non-blocking)
- YOLO detection (CUDA)
- VLM batch classification (async)
- Fusion YOLO + VLM (Bayesian fusion)
- Rendering real-time
"""

from __future__ import annotations
import cv2
import time
import threading
from queue import Queue

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
        self.vlm = VLMClassifier(cache_path="weights/vlm/text_features.pt")

        # Optimizer
        self.opt = ModelOptimizer()
        self.opt.warmup_yolo(self.detector.model, img_size=img_size)

        self.cap = cv2.VideoCapture(camera_id)
        self.running = True

    # --------------------------------------------------------------
    # CAMERA THREAD
    # --------------------------------------------------------------

    def camera_thread(self):
        print("🎥 Camera thread started")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    # --------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------

    def run(self):
        # Start camera thread
        cam_thread = threading.Thread(target=self.camera_thread, daemon=True)
        cam_thread.start()

        print("🔥 Real-time system started")

        pending_vlm = []

        while True:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()

            # YOLO detect
            dets = self.detector.detect(frame)

            # Collect crops for VLM
            crops_boxes = [d["bbox"] for d in dets]

            # Run VLM batch
            if self.show_vlm:
                vlm_results = self.vlm.classify_crops(frame, crops_boxes)
            else:
                vlm_results = []

            # Fusion + Drawing
            out_frame = self.render(frame, dets, vlm_results)

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

        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            yolo_conf = det["confidence"]

            color = (0, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {yolo_conf:.2f}"

            # VLM overlay
            if i < len(vlm_results):
                vlm = vlm_results[i]
                vlm_label = vlm["pred_label"]
                vlm_conf = vlm["pred_score"]

                combined = fusion_bayesian(yolo_conf, vlm_conf)
                label += f" | {vlm_label} {vlm_conf:.2f} | Fused {combined:.2f}"

            # draw label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return img
