import torch
import time
import os

class ModelOptimizer:
    def __init__(self):
        pass

    def warmup_yolo(self, model, img_size=320):
        print("Warming up YOLO ...")
        
        dummy_input = torch.zeros((1, 3, img_size, img_size))
        
        if torch.cuda.is_available():
            dummy_input = dummy_input.to('cuda')

        for _ in range(10):
            model.predict(source=dummy_input, verbose=False, device=0 if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        print("✅ YOLO warmup done")

    def benchmark(self, model_path, imgsz=320, num_iterations=100):
        
        print(f"Benchmarking {model_path} ...")
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        
        model.benchmark(imgsz=imgsz, device=0 if torch.cuda.is_available() else 'cpu')

    def export_to_trt(self, model_path, output):
        """
        Hàm wrapper để export TensorRT (chỉ chạy khi có GPU)
        """
        if not torch.cuda.is_available():
            print("Error: TensorRT export requires a GPU.")
            return

        print(f"Exporting {model_path} to TensorRT...")
        from ultralytics import YOLO
        model = YOLO(model_path)
        model.export(format="engine", device=0)