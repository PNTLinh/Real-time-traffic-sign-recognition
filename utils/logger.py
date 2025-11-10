from __future__ import annotations
import os, sys, time, math, logging
from typing import List, Sequence, Tuple, Optional
import numpy as np
import cv2


def setup_logger(
    name: str = "traffic-sign",
    save_dir: str = "outputs/logs",
    filename: str = "app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Khởi tạo logger ghi ra console + file (UTF-8).
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # tránh add handlers lặp
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(save_dir, filename), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    return logger

def _text_size(text: str, font_scale: float = 0.6, thickness: int = 1):
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg: bool = True,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
):
    """
    Vẽ text có nền (dễ đọc trên video).
    """
    (w, h) = _text_size(text, font_scale, thickness)
    x, y = org
    if bg:
        pad = 4
        cv2.rectangle(img, (x - pad, y - h - pad), (x + w + pad, y + pad), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _color_for_id(idx: int) -> Tuple[int, int, int]:
    """
    Tạo màu ổn định theo id lớp.
    """
    np.random.seed((idx * 9973) % 2**32)
    c = np.random.randint(50, 230, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def draw_boxes(
    img: np.ndarray,
    boxes_xyxy: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Optional[Sequence[float]] = None,
    class_ids: Optional[Sequence[int]] = None,
    show_score: bool = True,
) -> np.ndarray:
    """
    Vẽ bbox + nhãn (dùng cho YOLO hoặc nhãn đã fuse YOLO×VLM).
    """
    h, w = img.shape[:2]
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        if class_ids is not None:
            color = _color_for_id(int(class_ids[i]))
        else:
            color = (0, 200, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = str(labels[i]) if i < len(labels) else ""
        sc = f"{scores[i]:.2f}" if (scores is not None and i < len(scores)) else ""
        caption = f"{name} {sc}" if (show_score and sc) else name
        put_text(img, caption, (x1 + 2, max(20, y1 - 6)), font_scale=0.55, color=(255, 255, 255), bg=True, bg_color=color)
    return img


class FPSMeter:
    """
    Tính FPS trung bình trượt.
    """
    def __init__(self, avg_over: int = 30):
        self.avg_over = max(1, int(avg_over))
        self.ts: List[float] = []

    def update(self):
        now = time.time()
        self.ts.append(now)
        if len(self.ts) > self.avg_over:
            self.ts.pop(0)

    @property
    def fps(self) -> float:
        if len(self.ts) < 2:
            return 0.0
        dt = self.ts[-1] - self.ts[0]
        return 0.0 if dt <= 0 else (len(self.ts) - 1) / dt


def overlay_fps(img: np.ndarray, fps: float, org=(10, 24)):
    put_text(img, f"FPS: {fps:.1f}", org, font_scale=0.7, color=(255, 255, 255), bg=True, bg_color=(40, 40, 40))
