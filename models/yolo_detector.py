import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional

class YOLODetector:
    """
    Clean-refactor YOLO detector:
    - RGB input (fix accuracy)
    - Safe device placement
    - Clean outputs
    - Batch detection
    """

    def __init__(
        self,
        model_path="weights/yolo/best.pt",
        conf=0.5,
        iou=0.45,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Loading YOLO model on {self.device} ...")

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf = conf
        self.iou = iou
        self.class_names = self.model.names
        print(f"📊 Classes: {len(self.class_names)}")

    # -----------------------------------------------------------------
    # DETECT SINGLE IMAGE
    # -----------------------------------------------------------------

    def detect(self, image_bgr) -> List[Dict]:
        """Detect on one frame."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        result = self.model.predict(
            rgb,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
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

    # -----------------------------------------------------------------
    # BATCH DETECT
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------------------

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
