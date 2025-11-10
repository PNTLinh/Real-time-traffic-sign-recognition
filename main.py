# main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse, yaml
from typing import List, Sequence, Tuple

import numpy as np
import cv2

from utils.logger import setup_logger, draw_boxes, FPSMeter, overlay_fps

# ---- YOLO detector (bạn đã viết) ----
# Yêu cầu: phải có lớp YoloDetector với:
#   - thuộc tính class_names: List[str]
#   - predict(frame_bgr) -> (boxes_xyxy, scores, class_ids)
try:
    from models.yolo_detector import YoloDetector  # type: ignore
except Exception as e:
    raise ImportError(
        "Không tìm thấy models/yolo_detector.py với lớp YoloDetector. "
        "Hãy đảm bảo bạn đã implement detector này."
    ) from e

# ---- VLM ----
from models.vlm_classifier import VLMClassifier


def load_vlm_from_config(cfg_vlm: dict) -> VLMClassifier:
    with open(cfg_vlm["config_json"], "r", encoding="utf-8") as f:
        vcfg = json.load(f)

    # đọc prompts từ file
    templates = []
    prompt_file = vcfg.get("prompt_file")
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as pf:
            for line in pf:
                line = line.strip()
                if line:
                    templates.append(line)

    vlm = VLMClassifier(
        model_name=vcfg.get("model_name", "ViT-B-32"),
        pretrained=vcfg.get("pretrained", "laion2b_s34b_b79k"),
        use_half=bool(vcfg.get("use_half", True)),
        labels=vcfg.get("labels"),
        templates_vi=templates if templates else None,
        templates_en=[],
    )
    return vlm


def fuse_label(
    yolo_label: str,
    vlm_label: str,
    vlm_score: float,
    strategy: str = "consensus",
    threshold: float = 0.6,
) -> str:
    """
    Chiến lược ghép nhãn:
      - 'yolo_only'   : dùng YOLO
      - 'vlm_only'    : dùng VLM
      - 'and'         : "YOLO: X | VLM: Y(0.92)"
      - 'consensus'   : nếu score >= threshold -> dùng VLM; ngược lại dùng YOLO
    """
    strategy = (strategy or "consensus").lower()
    if strategy == "yolo_only":
        return yolo_label
    if strategy == "vlm_only":
        return vlm_label
    if strategy == "and":
        return f"{yolo_label} | {vlm_label}({vlm_score:.2f})"

    # consensus (default)
    return vlm_label if vlm_score >= threshold else yolo_label


def main(cfg_path: str):
    logger = setup_logger()

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    logger.info(f"Device: {device}")

    # --- YOLO ---
    yolo_cfg = cfg["yolo"]
    det = YoloDetector(
        weights=yolo_cfg["weights"],
        img_size=yolo_cfg.get("img_size", 640),
        conf_thres=yolo_cfg.get("conf_thres", 0.25),
        iou_thres=yolo_cfg.get("iou_thres", 0.45),
        max_det=yolo_cfg.get("max_det", 50),
        classes=yolo_cfg.get("classes", None),
        device=device,
    )
    logger.info("YOLO detector loaded.")

    # --- VLM ---
    vlm_cfg = cfg["vlm"]
    vlm_enabled = bool(vlm_cfg.get("enabled", True))
    vlm = None
    if vlm_enabled:
        vlm = load_vlm_from_config(vlm_cfg)
        logger.info("VLM classifier loaded.")
    fuse_strategy = vlm_cfg.get("fuse_strategy", "consensus")
    fuse_threshold = float(vlm_cfg.get("consensus_threshold", 0.6))
    min_crop_edge = int(vlm_cfg.get("min_crop_edge", 12))

    # --- Video IO ---
    vcfg = cfg["video"]
    cap = cv2.VideoCapture(vcfg.get("source", 0))
    if not cap.isOpened():
        logger.error("Không mở được nguồn video.")
        return

    save_out = bool(vcfg.get("save_out", False))
    writer = None
    if save_out:
        out_path = vcfg.get("out_path", "outputs/demo.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fps_w = vcfg.get("writer_fps", 25)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_w, (w, h)
        )
        logger.info(f"Ghi video ra: {out_path}")

    show = bool(vcfg.get("display", True))
    fps_meter = FPSMeter(avg_over=30)

    logger.info("Bắt đầu realtime… Nhấn 'q' để thoát.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- YOLO predict ---
        boxes, scores, cls_ids = det.predict(frame)  # bạn đã implement
        # lọc bbox quá nhỏ
        keep = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if min(x2 - x1, y2 - y1) >= min_crop_edge:
                keep.append(i)
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        cls_ids = [cls_ids[i] for i in keep]

        # --- Tên lớp từ YOLO ---
        yolo_names = getattr(det, "class_names", None)
        yolo_labels = [yolo_names[c] if (yolo_names and c < len(yolo_names)) else str(c) for c in cls_ids]

        # --- VLM classify các crop ---
        fused_labels = yolo_labels
        if vlm_enabled and vlm and len(boxes) > 0:
            outs = vlm.classify_detections(frame, boxes, topk=3)
            fused_labels = []
            for yl, out in zip(yolo_labels, outs):
                vl, vs = out["pred_label"], float(out["pred_score"])
                fused_labels.append(fuse_label(yl, vl, vs, fuse_strategy, fuse_threshold))

        # --- Vẽ ---
        draw_boxes(
            frame,
            boxes_xyxy=boxes,
            labels=fused_labels if cfg["draw"].get("show_vlm", True) else yolo_labels,
            scores=scores if cfg["draw"].get("show_yolo", True) else None,
            class_ids=cls_ids,
        )

        # --- FPS ---
        fps_meter.update()
        if cfg["draw"].get("show_fps", True):
            overlay_fps(frame, fps_meter.fps)

        if show:
            cv2.imshow("Traffic-Sign (YOLO + VLM)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Kết thúc.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn file config.yaml")
    args = parser.parse_args()
    main(args.config)
