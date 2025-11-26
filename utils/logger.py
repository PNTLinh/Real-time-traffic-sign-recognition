"""
Utility: Logging + Drawing Boxes + FPS Counter
Refactor version (an toàn hơn – hiệu suất cao – ổn định)
"""

from __future__ import annotations
import os, sys, time, logging
from typing import List, Tuple, Sequence, Optional
import numpy as np
import cv2


# =============================================================
# LOGGER
# =============================================================

def setup_logger(
    name: str = "traffic-sign",
    save_dir: str = "outputs/logs",
    filename: str = "app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Khởi tạo logger ghi ra console + file.
    Không add duplicate handlers.
    """
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # tránh nhân đôi handlers

    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    # File
    fh = logging.FileHandler(os.path.join(save_dir, filename), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    return logger


# =============================================================
# TEXT + DRAWING UTILITIES
# =============================================================

def _text_size(text: str, font_scale: float, thickness: int):
    return cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )[0]


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg: bool = True,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.65,
):
    """
    Vẽ text có nền mờ → dễ đọc. 
    - alpha: độ mờ của nền (0–1)
    """
    x, y = org
    (w, h) = _text_size(text, font_scale, thickness)

    if bg:
        pad = 4
        x1, y1 = x - pad, y - h - pad
        x2, y2 = x + w + pad, y + pad

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(
        img, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        color, thickness, cv2.LINE_AA
    )


# =============================================================
# COLOR PALETTE
# =============================================================

def _color_for_id(idx: int) -> Tuple[int, int, int]:
    """
    Tạo màu ổn định (fixed seed) theo class_id.
    """
    rng = np.random.default_rng(seed=idx * 123457)
    r, g, b = rng.integers(50, 215, size=3)
    return int(r), int(g), int(b)


# =============================================================
# DRAW BOUNDING BOX
# =============================================================

def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    class_ids: Optional[Sequence[int]] = None,
    show_score: bool = True,
) -> np.ndarray:
    """
    Vẽ bounding boxes + nhãn + điểm confidence.
    Hỗ trợ YOLO-only, VLM-only hoặc fused-class.
    """
    h, w = img.shape[:2]

    for i, box in enumerate(boxes_xyxy):
        # Clip & ensure int
        x1 = max(0, min(w - 1, int(box[0])))
        y1 = max(0, min(h - 1, int(box[1])))
        x2 = max(0, min(w - 1, int(box[2])))
        y2 = max(0, min(h - 1, int(box[3])))

        # Stable color
        color = (
            _color_for_id(class_ids[i])
            if class_ids is not None
            else (0, 200, 0)
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        name = labels[i] if i < len(labels) else ""
        sc = f"{scores[i]:.2f}" if (scores and i < len(scores)) else ""
        caption = f"{name} {sc}" if (show_score and sc) else name

        # Insert text
        put_text(
            img,
            caption,
            org=(x1 + 3, y1 - 6),
            font_scale=0.55,
            color=(255, 255, 255),
            thickness=1,
            bg=True,
            bg_color=color,
            alpha=0.55,
        )

    return img


# =============================================================
# FPS METER
# =============================================================

class FPSMeter:
    """
    Tính FPS trung bình bằng sliding window.
    """
    def __init__(self, avg_over: int = 30):
        self.avg_over = max(2, avg_over)
        self.timestamps: List[float] = []

    def update(self):
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) > self.avg_over:
            self.timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / dt if dt > 0 else 0.0


def overlay_fps(
    img: np.ndarray,
    fps: float,
    org: Tuple[int, int] = (10, 25),
):
    put_text(
        img,
        f"FPS: {fps:.1f}",
        org=org,
        font_scale=0.7,
        color=(255, 255, 255),
        bg=True,
        bg_color=(40, 40, 40),
        thickness=2,
    )
