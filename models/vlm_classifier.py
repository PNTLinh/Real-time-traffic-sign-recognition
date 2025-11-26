from __future__ import annotations
import os
from typing import List, Tuple, Dict, Sequence, Optional
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

try:
    import open_clip
except Exception:
    raise ImportError("Cần cài: pip install open-clip-torch")

LABELS_VN = [
    "Cấm đỗ xe", "Cấm dừng đỗ xe", "Cấm ngược chiều", "Cấm ô tô",
    "Cấm quay đầu", "Cấm rẽ phải", "Cấm rẽ trái", "Dừng lại",
    "Đường không bằng phẳng", "Đường không ưu tiên", "Đường ưu tiên",
    "Người đi bộ", "Tốc độ 30", "Tốc độ 40", "Tốc độ 50",
    "Tốc độ 60", "Tốc độ 80", "Trẻ em qua đường", "Vòng xuyến"
]

TEMPLATES_VI = [
    "biển báo giao thông: {}",
    "biển báo đường bộ Việt Nam: {}",
    "ảnh chụp biển báo: {}",
]

TEMPLATES_EN = [
    "traffic sign: {}",
    "road sign: {}",
    "photo of a traffic sign: {}",
]
class VLMClassifier:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        use_half: bool = True,
        labels: Optional[List[str]] = None,
        templates_vi: Optional[List[str]] = None,
        templates_en: Optional[List[str]] = None,
        cache_path: str = "weights/vlm/text_features.pt",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading OpenCLIP model: {model_name} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.use_half = False
        if use_half and self.device.startswith("cuda"):
            try:
                self.model = self.model.half()
                self.use_half = True
                print("Using FP16 (safe-mode enabled)")
            except Exception:
                print("FP16 not supported, fallback to FP32")

        self.labels = labels or LABELS_VN
        self.templates_vi = templates_vi or TEMPLATES_VI
        self.templates_en = templates_en or TEMPLATES_EN

        self.cache_path = Path(cache_path)
        self.text_features: Optional[torch.Tensor] = None

        self._init_text_features()

        print(f"VLM Classifier ready on {self.device}")
        print(f"Classes: {len(self.labels)}")

    def _init_text_features(self):
        """Load từ cache hoặc build mới."""
        if self.cache_path.exists():
            print("Loading cached text features ...")
            self.text_features = torch.load(self.cache_path, map_location=self.device)
            return

        print("Building text features ...")
        self._build_text_features()
        self._save_cache()

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.text_features, self.cache_path)
        print(f"Text features cached → {self.cache_path}")

    def _build_text_features(self):
        prompts = []
        for name in self.labels:
            for t in self.templates_vi:
                prompts.append(t.format(name))
            for t in self.templates_en:
                prompts.append(t.format(name))

        tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        k = len(self.templates_vi) + len(self.templates_en)
        per_class = []
        for i in range(len(self.labels)):
            per_class.append(feats[i * k: (i + 1) * k].mean(dim=0))

        self.text_features = torch.stack(per_class)

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        t = self.preprocess(image).unsqueeze(0).to(self.device)
        return t.half() if self.use_half else t

    @torch.inference_mode()
    def classify_image(self, image: Image.Image, topk: int = 3) -> Dict:
        img = self._pil_to_tensor(image)
        img_feat = self.model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        logits = img_feat @ self.text_features.T
        probs = logits.softmax(dim=-1)

        topk = min(topk, len(self.labels))
        scores, idxs = probs.topk(topk, dim=-1)

        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        return {
            "pred_label": self.labels[idxs[0]],
            "pred_score": scores[0],
            "topk_labels": [self.labels[i] for i in idxs],
            "topk_scores": scores,
            "logits": logits.cpu().numpy(),
        }

    def safe_crop(self, img: np.ndarray, x1, y1, x2, y2):
        h, w = img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:  
            return None

        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            return None

        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    @torch.inference_mode()
    def classify_crops(self, image_bgr: np.ndarray, boxes_xyxy: Sequence):
        crops = []
        for x1, y1, x2, y2 in boxes_xyxy:
            crop = self.safe_crop(image_bgr, x1, y1, x2, y2)
            if crop is not None:
                crops.append(crop)

        if not crops:
            return []

        batch = torch.cat([self._pil_to_tensor(im) for im in crops], dim=0)
        img_feat = self.model.encode_image(batch)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        logits = img_feat @ self.text_features.T
        probs = logits.softmax(dim=-1)

        out = []
        for i in range(len(crops)):
            scores, idxs = probs[i].topk(3)
            idxs = idxs.tolist()
            scores = scores.tolist()

            out.append({
                "pred_label": self.labels[idxs[0]],
                "pred_score": scores[0],
                "topk_labels": [self.labels[x] for x in idxs],
                "topk_scores": scores,
            })
        return out
