# evaluation/evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
import numpy as np
from PIL import Image

from .metrics import (
    DetGT, DetPred, detection_map,
    accuracy, precision_recall_f1, confusion_matrix
)

# ===== Adapters (bạn cắm model thật của mình vào đây) =====

class YOLOv12Detector:
    """Adapter đơn giản cho YOLOv12. Bạn thay .detect(...) bằng code thật."""
    def __init__(self, model_path: str, conf_thres: float = 0.25):
        self.conf_thres = conf_thres
        # TODO: load model tại đây

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Return: list[ { 'bbox':[x1,y1,x2,y2], 'score':float, 'cls':int } ]
        Toạ độ theo ảnh gốc.
        """
        # TODO: trả kết quả thật
        return []

class VLMClassifier:
    """Adapter cho VLM (phân loại/QA trên crop)."""
    def __init__(self, prompt: str = "Classify this object"):
        self.prompt = prompt
        # TODO: init VLM

    def classify(self, crop: Image.Image) -> Dict[str, Any]:
        """
        Return: {'label': int, 'score': float, 'text': 'optional answer'}
        """
        # TODO: trả kết quả thật
        return {"label": 0, "score": 1.0, "text": ""}

# ===== Dataset interface =====

@dataclass
class Sample:
    image_id: str
    image_path: str
    gt_boxes: np.ndarray       # [N,4] xyxy
    gt_labels: List[int]       # [N]
    # optional: ground-truth text/answers per box or per image if dùng VQA

class SimpleFolderDataset:
    """
    Ví dụ dataset đơn giản.
    Bạn tự viết loader thật: đọc từ COCO-style JSON, CSV, hay folder riêng.
    """
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __iter__(self):
        for s in self.samples:
            yield s

# ===== Evaluator =====

class Evaluator:
    def __init__(
        self,
        detector: YOLOv12Detector,
        vlm: VLMClassifier,
        class_names: List[str],
        iou_thresholds: Iterable[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        det_conf_thres: float = 0.25
    ):
        self.detector = detector
        self.vlm = vlm
        self.class_names = class_names
        self.iou_thresholds = list(iou_thresholds)
        self.det_conf_thres = det_conf_thres

    @staticmethod
    def _crop(image: Image.Image, box: List[float]) -> Image.Image:
        x1,y1,x2,y2 = [int(round(v)) for v in box]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.width-1, x2); y2 = min(image.height-1, y2)
        return image.crop((x1,y1,x2,y2))

    def evaluate(self, dataset: SimpleFolderDataset, save_jsonl: Optional[str] = None) -> Dict[str, Any]:
        """
        Chạy pipeline YOLO → VLM end-to-end và tính:
        - Detection: mAP@[.50:.95]
        - Classification: accuracy, macro/micro precision/recall/f1, confusion matrix
        """
        gt_list: List[DetGT] = []
        pred_list: List[DetPred] = []
        cls_labels_all: List[int] = []
        cls_preds_all: List[int] = []

        import json
        jf = open(save_jsonl, "w", encoding="utf-8") if save_jsonl else None

        for sample in dataset:
            img = Image.open(sample.image_path).convert("RGB")
            dets = self.detector.detect(img)  # [{'bbox':..., 'score':..., 'cls':...}, ...]

            # lọc theo conf
            dets = [d for d in dets if d.get("score", 0.0) >= self.det_conf_thres]

            # VLM phân loại từng bbox (nếu YOLO không có nhãn hoặc muốn refine)
            pred_boxes, pred_scores, pred_labels = [], [], []
            for d in dets:
                crop = self._crop(img, d["bbox"])
                cls_res = self.vlm.classify(crop)
                label = int(cls_res.get("label", d.get("cls", -1)))
                score = float(min(max(cls_res.get("score", d.get("score", 0.0)), 0.0), 1.0))
                pred_boxes.append(d["bbox"])
                pred_scores.append(score)
                pred_labels.append(label)

                # thu thập cho classification metrics nếu có GT mapping 1-1 theo box
                # (ở đây đơn giản: ghép theo IOU max > 0.5 với GT để lấy label thật)
            if len(pred_boxes) == 0:
                pred_boxes = np.zeros((0,4), dtype=np.float32)
                pred_scores = np.zeros((0,), dtype=np.float32)
            else:
                pred_boxes = np.array(pred_boxes, dtype=np.float32)
                pred_scores = np.array(pred_scores, dtype=np.float32)

            # Push detection GT/PRED
            gt_list.append(DetGT(sample.image_id, sample.gt_boxes, sample.gt_labels))
            pred_list.append(DetPred(sample.image_id, pred_boxes, pred_scores, pred_labels))

            # Gom nhãn cho classification accuracy bằng cách match gần nhất IOU>=0.5
            if len(sample.gt_boxes) and len(pred_boxes):
                from .metrics import box_iou_xyxy
                iou_mat = box_iou_xyxy(sample.gt_boxes, pred_boxes)
                for gi in range(iou_mat.shape[0]):
                    pj = int(np.argmax(iou_mat[gi]))
                    if iou_mat[gi, pj] >= 0.5:
                        cls_labels_all.append(int(sample.gt_labels[gi]))
                        cls_preds_all.append(int(pred_labels[pj]))

            if jf:
                jf.write(json.dumps({
                    "image_id": sample.image_id,
                    "pred": [{"bbox": b, "score": float(s), "label": int(l)} for b,s,l in zip(pred_boxes.tolist(), pred_scores.tolist(), pred_labels)],
                    "gt": [{"bbox": b, "label": int(l)} for b,l in zip(sample.gt_boxes.tolist(), sample.gt_labels)]
                }, ensure_ascii=False) + "\n")

        if jf: jf.close()

        # ---- Metrics ----
        det_report = detection_map(gt_list, pred_list, class_ids=range(len(self.class_names)), iou_thresholds=self.iou_thresholds)

        cls_report = {}
        if cls_labels_all:
            cls_report["accuracy"] = accuracy(cls_labels_all, cls_preds_all)
            cls_report.update(precision_recall_f1(cls_labels_all, cls_preds_all, num_classes=len(self.class_names)))
            cls_report["confusion_matrix"] = confusion_matrix(cls_labels_all, cls_preds_all, len(self.class_names)).tolist()

        report = {"detection": det_report, "classification": cls_report, "num_images": len(dataset)}
        print(json_pretty(report))
        return report

def json_pretty(o: Any) -> str:
    import json
    return json.dumps(o, indent=2, ensure_ascii=False)
