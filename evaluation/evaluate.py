"""
Đánh giá VLM trên các crop GT theo định dạng nhãn YOLO:
mỗi ảnh có file .txt: <class_id> <x_center> <y_center> <w> <h> (chuẩn hoá 0..1)
Script sẽ crop vùng GT, chạy VLM zero-shot, rồi in Accuracy/Top-k/F1.
"""
from __future__ import annotations
import os, glob
from typing import List, Tuple
import numpy as np
from PIL import Image
import cv2

from models.vlm_classifier import VLMClassifier
from evaluation.metrics import evaluate_classification


def load_yolo_label(label_path: str, W: int, H: int) -> List[Tuple[int, List[int]]]:
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)
            boxes.append((cid, [x1, y1, x2, y2]))
    return boxes


def main(images_dir: str):
    vlm = VLMClassifier()  # ViT-B/32, laion2b
    y_true, logits_all = [], []

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join(images_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(images_dir, "**", "*.jpeg"), recursive=True)
    )
    for img_path in image_paths:
        lab_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(lab_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        items = load_yolo_label(lab_path, W, H)
        if not items:
            continue

        crops = []
        gts = []
        pil_rgb = Image.fromarray(img[:, :, ::-1])
        for cid, (x1, y1, x2, y2) in items:
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(pil_rgb.crop((x1, y1, x2, y2)))
            gts.append(cid)

        if not crops:
            continue

        outs = vlm.batched_predict(crops, topk=5)
        y_true.extend(gts)
        logits_all.extend([o["logits"] for o in outs])

    if len(y_true) == 0:
        print("Không có dữ liệu hợp lệ.")
        return

    logits = np.stack(logits_all, axis=0)
    report = evaluate_classification(logits, y_true, ks=(1, 3, 5))
    print(f"Samples: {report['support']}")
    print(f"Accuracy: {report['accuracy']:.4f} | Top1: {report['top1']:.4f} | Top3: {report['top3']:.4f} | Top5: {report['top5']:.4f}")
    print(f"Macro F1: {report['macro_f1']:.4f} | Micro F1: {report['micro_f1']:.4f} | Weighted F1: {report['weighted_f1']:.4f}")
    # Nếu cần: print(report['per_class']) hoặc lưu confusion_matrix


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/val")
