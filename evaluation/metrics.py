# evaluation/metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import numpy as np


# ============ Classification metrics ============
def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    return float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0


def topk_accuracy(
    logits: np.ndarray, y_true: Sequence[int], ks: Sequence[int] = (1, 3, 5)
) -> Dict[str, float]:
    """
    logits: (N, C) — từ VLM (similarity/logits)
    y_true: (N,)
    """
    logits = np.asarray(logits)
    y_true = np.asarray(y_true, dtype=np.int64)
    assert logits.ndim == 2 and logits.shape[0] == y_true.shape[0]
    order = np.argsort(-logits, axis=1)  # desc
    out = {}
    for k in ks:
        k = min(k, logits.shape[1])
        hits = (order[:, :k] == y_true[:, None]).any(axis=1)
        out[f"top{k}"] = float(hits.mean())
    return out


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: Optional[int] = None
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1 if y_true.size else 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: Optional[int] = None
) -> Dict[str, object]:
    """
    Trả về:
      - per_class: dict {i: {"precision":..., "recall":..., "f1":..., "support": int}}
      - macro_precision/recall/f1
      - micro_precision/recall/f1
      - weighted_f1
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    support = cm.sum(axis=1)

    precision = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
    recall = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    per_class = {
        int(i): {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(TP))
    }

    macro_p = float(np.nanmean(precision)) if precision.size else 0.0
    macro_r = float(np.nanmean(recall)) if recall.size else 0.0
    macro_f1 = float(np.nanmean(f1)) if f1.size else 0.0

    # micro: tính từ tổng TP/FP/FN
    TP_sum = float(TP.sum())
    FP_sum = float(FP.sum())
    FN_sum = float(FN.sum())
    micro_p = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
    micro_r = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    )

    # weighted f1 theo support
    total = float(support.sum()) if support.size else 0.0
    weighted_f1 = (
        float((f1 * support).sum() / total) if total > 0 else 0.0
    )

    return {
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "support": int(total),
        "confusion_matrix": cm,
    }


def evaluate_classification(
    logits: np.ndarray, y_true: Sequence[int], ks: Sequence[int] = (1, 3, 5)
) -> Dict[str, object]:
    """
    Gói nhanh: từ logits (N,C) + nhãn thật → bộ chỉ số đầy đủ.
    """
    logits = np.asarray(logits)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = logits.argmax(axis=1)
    out = {"accuracy": accuracy(y_true, y_pred)}
    out.update(topk_accuracy(logits, y_true, ks=ks))
    out.update(precision_recall_f1(y_true, y_pred))
    return out


# ============ Retrieval (tuỳ chọn) ============
def retrieval_recall_at_k(sim: np.ndarray, positives: List[Sequence[int]],
                          ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """
    sim: (Q, C) điểm tương đồng query→candidate (càng lớn càng tốt).
    positives[q]: index ứng viên đúng của query q.
    """
    Q, C = sim.shape
    assert len(positives) == Q
    order = np.argsort(-sim, axis=1)
    ranks = []
    for q in range(Q):
        pos = set(int(i) for i in positives[q])
        r = C + 1
        for rank, idx in enumerate(order[q], start=1):
            if idx in pos:
                r = rank
                break
        ranks.append(r)
    out = {f"R@{k}": float(np.mean([1.0 if r <= k else 0.0 for r in ranks])) for k in ks}
    out["MedR"] = float(np.median(ranks))
    out["MeanR"] = float(np.mean(ranks))
    return out
