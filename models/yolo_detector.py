from ultralytics import YOLO
import numpy as np

class YoloDetector:
    def __init__(
        self,
        weights: str,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 50,
        classes=None,
        device: str = "cpu",
    ):
        self.model = YOLO(weights)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.device = device

        # Load class names from the model's YOLO data
        data = self.model.model.yaml.get('names', None) if hasattr(self.model, 'model') else None
        if data is None:
            self.class_names = []
        elif isinstance(data, dict):
            self.class_names = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
        else:
            self.class_names = list(data)

    def predict(self, frame_bgr):
        # Run prediction on BGR frame
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            verbose=False,
            device=self.device,
        )
        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return [], [], []

        xyxy = boxes.xyxy.cpu().numpy().tolist()
        scores = boxes.conf.cpu().numpy().tolist()
        cls_ids = boxes.cls.cpu().numpy().astype(int).tolist()

        return xyxy, scores, cls_ids
