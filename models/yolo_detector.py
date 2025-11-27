import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.tasks 
from typing import List, Dict, Optional

class YOLODetector:
    def __init__(
        self,
        model_path="weights/yolo/best.pt",
        conf=0.1,
        iou=0.6,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading YOLO model from: {model_path}")
        safe_list = [ultralytics.nn.tasks.DetectionModel]
        for name in dir(nn):
            attr = getattr(nn, name)
            if isinstance(attr, type):
                safe_list.append(attr)
        torch.serialization.add_safe_globals(safe_list)
        self.model = YOLO(model_path, task='detect')
        if model_path.endswith('.pt'):
            torch_device = f"cuda:{self.device}" if isinstance(self.device, int) else self.device
            self.model.to(torch_device)
            print(f"PyTorch model moved to {torch_device}")
        else:
            print(f"Optimized model loaded (TensorRT/ONNX). Skipping .to() move.")

        self.conf = conf
        self.iou = iou
        self.class_names = self.model.names
        print(f"Classes: {len(self.class_names)}")

    def detect(self, image_bgr) -> List[Dict]:
        """Detect on one frame."""
        # Ultralytics can handle BGR directly, but converting to RGB is safer practice
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        result = self.model.predict(
            rgb,
            conf=self.conf,
            iou=self.iou,
            device=self.device, # Pass 0 for GPU, 'cpu' for CPU
            verbose=False
        )[0]

        dets = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            cls = int(box.cls[0])
            dets.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class_id": cls,
                "class_name": self.class_names[cls],
                "confidence": float(box.conf[0]),
            })
        return dets

    def detect_batch(self, images_bgr: List):
        rbg_list = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
        
        results = self.model.predict(
            rbg_list,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        batch_out = []
        for result in results:
            dets = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                cls = int(box.cls[0])
                dets.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls,
                    "class_name": self.class_names[cls],
                    "confidence": float(box.conf[0]),
                })
            batch_out.append(dets)
        return batch_out

    def visualize(self, image_bgr, detections):
        img = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls = det["class_name"]
            conf = det["confidence"]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{cls} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img