from __future__ import annotations
import torch
from ultralytics import YOLO
import time
import os
from typing import Optional
import numpy as np


class ModelOptimizer:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def warmup_yolo(self, model: YOLO, img_size: int = 320, steps: int = 10):
        print("Warming up YOLO ...")
        dummy = torch.zeros(1, 3, img_size, img_size).to(self.device)
        for _ in range(steps):
            model(dummy)
        torch.cuda.synchronize()
        print("YOLO warmup done")

    def export_to_trt(self, model_path: str, output="./weights/yolo/best_trt.engine"):
        print("Exporting YOLO to TensorRT...")
        model = YOLO(model_path)
        engine_path = model.export(format="engine")
        print(f"TensorRT engine saved at: {engine_path}")
        return engine_path

    def create_cuda_graph(self, model: YOLO, img_size=320):
        if self.device != "cuda":
            return None

        print("Creating CUDA Graph for YOLO...")
        dummy = torch.zeros(1, 3, img_size, img_size, device="cuda")

        stream = torch.cuda.Stream()
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            for _ in range(3):  
                _ = model(dummy)

        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(dummy)

        print("CUDA Graph ready!")
        return g, dummy, out

    def enable_mixed_precision(self):
        return torch.cuda.amp.autocast(enabled=True)

    def batch_scheduler(self, pending_list, max_batch=4):
        if len(pending_list) >= max_batch:
            batch = pending_list[:max_batch]
            del pending_list[:max_batch]
            return batch
        return None
