# main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse, yaml
from typing import List, Sequence, Tuple, Optional, Union

import numpy as np
import cv2
import torch

from utils.logger import setup_logger, draw_boxes, FPSMeter, overlay_fps
from utils.paths import PATHS

try:
    from models.yolo_detector import YoloDetector  # type: ignore
except Exception as e:
    raise ImportError(
        "Không tìm thấy models/yolo_detector.py với lớp YoloDetector. "
    ) from e

# ---- VLM ----
from models.vlm_classifier import VLMClassifier


def _res(pathlike: Union[str, int, None]) -> Union[str, int, None]:
    """Chuẩn hoá backslash và resolve alias local_root/... , local/..."""
    if isinstance(pathlike, str):
        pathlike = pathlike.replace("\\", "/")
    return PATHS.resolve(pathlike)


def load_vlm_from_config(cfg_vlm: dict) -> VLMClassifier:
    """
    Đọc JSON cấu hình VLM, hỗ trợ alias local_root/... và local/...
    """
    cfg_json_path = _res(cfg_vlm["config_json"])
    with open(cfg_json_path, "r", encoding="utf-8") as f:
        vcfg = json.load(f)

    # đọc prompts từ file (nếu có)
    templates: List[str] = []
    prompt_file = vcfg.get("prompt_file")
    if prompt_file:
        prompt_path = _res(prompt_file)
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as pf:
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
    return vlm_label if vlm_score >= threshold else yolo_label  # consensus


def main(cfg_path: str):
    logger = setup_logger()

    # ---- đọc config, rồi khởi tạo PATHS theo local_root ----
    cfg_path = _res(cfg_path)  # cho phép alias cả ở --config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Gốc cục bộ cho alias local_root/... và local/...
    local_root = (cfg.get("paths") or {}).get("local_root", ".")
    PATHS.init(local_root)

    # Resolve các path quan trọng
    vcfg = cfg["video"]
    vcfg["source"]   = _res(vcfg.get("source", 0))
    vcfg["out_path"] = _res(vcfg.get("out_path", "outputs/demo.mp4"))

    yolo_cfg = cfg["yolo"]
    yolo_cfg["weights"] = _res(yolo_cfg["weights"])

    vlm_cfg = cfg["vlm"]
    vlm_cfg["config_json"] = _res(vlm_cfg["config_json"])

    # ---- device ----
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Torch {torch.__version__} | CUDA {torch.version.cuda}")

    # --- YOLO ---
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
    vlm_enabled = bool(vlm_cfg.get("enabled", True))
    vlm: Optional[VLMClassifier] = None
    if vlm_enabled:
        vlm = load_vlm_from_config(vlm_cfg)
        logger.info("VLM classifier loaded.")
        # warm-up nhẹ để ổn định tốc độ
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = vlm.classify_detections(dummy, [[0, 0, 64, 64]], topk=1)

    fuse_strategy = vlm_cfg.get("fuse_strategy", "consensus")
    fuse_threshold = float(vlm_cfg.get("consensus_threshold", 0.6))
    min_crop_edge = int(vlm_cfg.get("min_crop_edge", 12))

    # --- Video IO ---
    cap = cv2.VideoCapture(vcfg.get("source", 0))
    if not cap.isOpened():
        logger.error("Không mở được nguồn video.")
        return

    save_out = bool(vcfg.get("save_out", False))
    writer = None
    if save_out:
        out_path = vcfg.get("out_path", "outputs/demo.mp4")
        PATHS.ensure_parent_dir(out_path)
        fps_w = vcfg.get("writer_fps", 25)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # codec fallback
        for cc in ("mp4v", "XVID", "avc1"):
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps_w, (w, h))
            if writer.isOpened():
                logger.info(f"Ghi video ra: {out_path} (codec {cc})")
                break
        if writer is None or not writer.isOpened():
            logger.warning("Không mở được VideoWriter; tắt save_out.")
            writer = None

    show = bool(vcfg.get("display", True))
    fps_meter = FPSMeter(avg_over=30)

    # Cảnh báo nếu danh sách lớp YOLO ≠ VLM (tránh fuse sai)
    try:
        if hasattr(det, "class_names") and det.class_names and vlm and vlm.labels:
            if set(det.class_names) != set(vlm.labels):
                logger.warning("Danh sách lớp YOLO khác VLM; hãy đồng bộ label để fuse chính xác.")
    except Exception:
        pass

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
    # dùng forward slash; có thể truyền alias: --config local_root/configs/config.yaml
    parser.add_argument("--config", default="configs/config.yaml", help="Đường dẫn file config.yaml")
    args = parser.parse_args()

    cfg_path = args.config
    main(cfg_path)
