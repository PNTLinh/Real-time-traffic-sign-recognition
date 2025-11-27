from __future__ import annotations
import cv2
import time
import threading
import os
import sys
from queue import Queue

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import numpy as np
from models.yolo_detector import YOLODetector
from models.vlm_classifier import VLMClassifier
from pipeline.optimizer import ModelOptimizer

def fusion_bayesian(p_yolo: float, p_vlm: float):
    return (p_yolo * p_vlm) / ((p_yolo * p_vlm) + (1 - p_yolo) * (1 - p_vlm) + 1e-8)
class RealTimeSystem:
    def __init__(
        self,
        model_path="weights/yolo/best.pt", 
        camera_id=0,
        img_size=320,
        show_vlm=True,
        batch_vlm=True,
    ):
        self.img_size = img_size
        self.show_vlm = show_vlm
        self.batch_vlm = batch_vlm

        self.frame_queue = Queue(maxsize=3)

        print(f"Initializing models from: {model_path}") 

        if not os.path.exists(model_path):
             print(f"ERROR: Model file not found at: {model_path}")

        self.detector = YOLODetector(model_path)

        vlm_cache = "weights/vlm/text_features.pt"
        if not os.path.exists(vlm_cache):
             print(f"Warning: VLM cache not found at {vlm_cache}. It will be created.")
        
        self.vlm = VLMClassifier(cache_path=vlm_cache)

        self.opt = ModelOptimizer()
        self.opt.warmup_yolo(self.detector.model, img_size=img_size)

        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.running = False

    def process_frame(self, frame):
    
        dets = self.detector.detect(frame)

        crops_boxes = [d["bbox"] for d in dets]

        if self.show_vlm and len(crops_boxes) > 0:
            vlm_results = self.vlm.classify_crops(frame, crops_boxes)
        else:
            vlm_results = []

        out_frame = self.render(frame, dets, vlm_results)
        return out_frame

    def run_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        print(f"Processing Image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print("Error: Could not read image.")
            return

        start_t = time.time()
        result_frame = self.process_frame(frame)
        end_t = time.time()

        print(f"Processed in {end_t - start_t:.4f}s")
        
        if IN_COLAB:
            print("Result:")
            cv2_imshow(result_frame)
        
        out_path = "output_" + os.path.basename(image_path)
        cv2.imwrite(out_path, result_frame)
        print(f"Saved to: {out_path}")

    def run_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"Error: Video not found at {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_name = "output_" + os.path.basename(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))

        print(f"Processing Video: {video_path} ({width}x{height} @ {fps}fps)")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            
            out.write(processed_frame)

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"⏳ Processing frame {frame_idx}/{total_frames}...", end='\r')

        cap.release()
        out.release()
        print(f"\nVideo processing complete! Saved to {out_name}")

    def render(self, frame, dets, vlm_results):
        img = frame.copy()

        limit = min(len(dets), len(vlm_results)) if vlm_results else 0

        for i, det in enumerate(dets):
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            yolo_conf = det["confidence"]

            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{cls} {yolo_conf:.2f}"

            if i < limit:
                vlm = vlm_results[i]
                vlm_label = vlm["pred_label"]
                vlm_conf = vlm["pred_score"]

                combined = fusion_bayesian(yolo_conf, vlm_conf)
                
                label = f"{vlm_label} | Fused: {combined:.2f}"
                
                if combined < 0.4:
                    color = (0, 0, 255) 
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return img