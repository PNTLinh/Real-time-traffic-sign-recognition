from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
import numpy as np
from collections import defaultdict, Counter

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận IoU giữa hai tập bounding box dạng (x1,y1,x2,y2).
    a: [Na,4] (x1,y1,x2,y2), b: [Nb,4]
    return: [Na,Nb] IoU matrix
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    area_a = (a[:,2]-a[:,0]).clip(0) * (a[:,3]-a[:,1]).clip(0)
    area_b = (b[:,2]-b[:,0]).clip(0) * (b[:,3]-b[:,1]).clip(0)

    lt = np.maximum(a[:,None,:2], b[None,:,:2])  # [Na,Nb,2]
    rb = np.minimum(a[:,None,2:], b[None,2:])    # [Na,Nb,2]
    wh = (rb - lt).clip(0)
    inter = wh[...,0] * wh[...,1] # tính phần giao(inter)
    union = area_a[:,None] + area_b[None,:] - inter # tính phần hợp(union)
    iou = np.where(union > 0, inter/union, 0)
    return iou

@dataclass
class DetGT:
    """ground-truth theo từng ảnh."""
    image_id: str
    boxes: np.ndarray        # [N,4] xyxy
    labels: List[int]        # class ids

@dataclass
class DetPred:
    """dự đoán theo từng ảnh."""
    image_id: str
    boxes: np.ndarray        # [M,4] xyxy
    scores: np.ndarray       # [M]
    labels: List[int]        # predicted class ids

def _ap_per_class(
    gts: List[DetGT],
    preds: List[DetPred],
    cls_id: int,
    iou_thresh: float = 0.5
) -> float:
    # gom tất cả GT/PRED của class này
    gt_boxes = []
    gt_imgids = []
    for g in gts:
        m = [i for i, c in enumerate(g.labels) if c == cls_id]
        if m:
            gt_boxes.append(g.boxes[m])
            gt_imgids += [g.image_id] * len(m)
    if gt_boxes:
        gt_boxes = np.concatenate(gt_boxes, axis=0)
    else:
        gt_boxes = np.zeros((0,4), dtype=np.float32)
    npos = len(gt_imgids)

    pred_boxes = []
    pred_scores = []
    pred_imgids = []
    for p in preds:
        m = [i for i, c in enumerate(p.labels) if c == cls_id]
        if m:
            pred_boxes.append(p.boxes[m])
            pred_scores.append(p.scores[m])
            pred_imgids += [p.image_id] * len(m)
    if pred_boxes:
        pred_boxes = np.concatenate(pred_boxes, axis=0)
        pred_scores = np.concatenate(pred_scores, axis=0)
    else:
        return 0.0  # không có prediction

    # sort theo confidence
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_imgids = [pred_imgids[i] for i in order]

    # tạo map GT theo image để match greedy
    gt_by_img: Dict[str, List[np.ndarray]] = defaultdict(list)
    used_by_img: Dict[str, List[bool]] = defaultdict(list)
    for g in gts:
        m = [i for i, c in enumerate(g.labels) if c == cls_id]
        if m:
            gt_by_img[g.image_id].append(g.boxes[m])
            used_by_img[g.image_id] += [False] * len(m)
    for k in list(gt_by_img.keys()):
        gt_by_img[k] = [gt_by_img[k][0]] if isinstance(gt_by_img[k], list) else gt_by_img[k]
        gt_by_img[k] = np.concatenate(gt_by_img[k], axis=0)

    tp = np.zeros(len(pred_boxes), dtype=np.float32)
    fp = np.zeros(len(pred_boxes), dtype=np.float32)
    for i, (pb, imgid) in enumerate(zip(pred_boxes, pred_imgids)):
        gtb = gt_by_img.get(imgid)
        if gtb is None or len(gtb) == 0:
            fp[i] = 1.0
            continue
        ious = box_iou_xyxy(pb[None,:], gtb)[0]  # [Ng]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thresh and not used_by_img[imgid][j]:
            tp[i] = 1.0
            used_by_img[imgid][j] = True
        else:
            fp[i] = 1.0

    # precision-recall
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(npos, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # AP = area dưới đường P-R (11-pt hoặc integrate)
    # dùng integration chuẩn
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for k in range(len(mpre)-1, 0, -1):
        mpre[k-1] = max(mpre[k-1], mpre[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))
    return ap

def detection_map(
    gts: List[DetGT],
    preds: List[DetPred],
    class_ids: Iterable[int],
    iou_thresholds: Iterable[float] = (0.5,)
) -> Dict[str, float]:
    """
    Trả về mAP@0.5 và/hoặc mAP@[0.5:0.95] tùy iou_thresholds.
    """
    class_ids = list(class_ids)
    iou_thresholds = list(iou_thresholds)
    aps = []
    for t in iou_thresholds:
        ap_t = []
        for c in class_ids:
            ap = _ap_per_class(gts, preds, c, iou_thresh=t)
            ap_t.append(ap)
        aps.append(np.mean(ap_t) if ap_t else 0.0)
    out = {}
    if len(iou_thresholds) == 1:
        out[f"mAP@{iou_thresholds[0]:.2f}"] = float(aps[0])
    else:
        out["mAP@[0.50:0.95]"] = float(np.mean(aps))
        for t, v in zip(iou_thresholds, aps):
            out[f"AP@{t:.2f}"] = float(v)
    return out

# ========== Classification / VQA metrics ==========

def accuracy(labels: List[int], preds: List[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(int(a == b) for a, b in zip(labels, preds))
    return correct / len(labels)

def confusion_matrix(labels: List[int], preds: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels, preds):
        if 0 <= y < num_classes and 0 <= p < num_classes:
            cm[y, p] += 1
    return cm

def precision_recall_f1(labels: List[int], preds: List[int], num_classes: int) -> Dict[str, float]:
    cm = confusion_matrix(labels, preds, num_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.where(tp+fp>0, tp/(tp+fp), 0.0)
    rec  = np.where(tp+fn>0, tp/(tp+fn), 0.0)
    f1   = np.where(prec+rec>0, 2*prec*rec/(prec+rec), 0.0)
    return {
        "precision_macro": float(np.mean(prec)),
        "recall_macro": float(np.mean(rec)),
        "f1_macro": float(np.mean(f1)),
        "precision_micro": float(tp.sum()/max((tp+fp).sum(),1)),
        "recall_micro": float(tp.sum()/max((tp+fn).sum(),1)),
        "f1_micro": float(2*tp.sum()/max(2*tp.sum()+fp.sum()+fn.sum(),1)),
    }

# VQA/Text simple metrics
def exact_match(ref: str, hyp: str) -> float:
    return float((ref or "").strip().lower() == (hyp or "").strip().lower())

def token_f1(ref: str, hyp: str) -> float:
    ref_t = (ref or "").lower().split()
    hyp_t = (hyp or "").lower().split()
    ref_c = Counter(ref_t)
    hyp_c = Counter(hyp_t)
    common = sum((ref_c & hyp_c).values())
    if common == 0:
        return 0.0
    prec = common / max(len(hyp_t), 1)
    rec  = common / max(len(ref_t), 1)
    return 2*prec*rec/(prec+rec)
