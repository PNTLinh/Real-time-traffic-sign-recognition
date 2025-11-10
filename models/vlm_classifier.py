from __future__ import annotations
import os
from typing import List, Tuple, Dict, Sequence, Optional

import torch
import numpy as np
from PIL import Image

try:
    import open_clip
except Exception as e:
    raise ImportError(
        "Yêu cầu 'open-clip-torch'. Cài đặt: pip install open-clip-torch"
    ) from e

VIETNAMESE_LABELS: List[str] = [
    "Cấm đỗ xe",
    "Cấm dừng đỗ xe",
    "Cấm ngược chiều",
    "Cấm ô tô",
    "Cấm quay đầu",
    "Cấm rẽ phải",
    "Cấm rẽ trái",
    "Dừng lại",
    "Đường không bằng phẳng",
    "Đường không ưu tiên",
    "Đường ưu tiên",
    "Người đi bộ",
    "Tốc độ 30",
    "Tốc độ 40",
    "Tốc độ 50",
    "Tốc độ 60",
    "Tốc độ 80",
    "Trẻ em qua đường",
    "Vòng xuyến",
]

DEFAULT_TEMPLATES_VI = [
    "biển báo giao thông: {}",
    "biển báo đường bộ Việt Nam: {}",
    "ảnh chụp biển báo: {}",
    "dấu hiệu giao thông: {}",
]
DEFAULT_TEMPLATES_EN = [
    "traffic sign: {}",
    "road sign: {}",
    "Vietnam traffic sign: {}",
    "photo of a traffic sign: {}",
]


class VLMClassifier:
    """
    Trình bao cho OpenCLIP: sinh text features từ label, so khớp cosine với image features.
    Hỗ trợ:
      - set_labels() để đổi danh sách lớp
      - classify_image() cho 1 ảnh/crop
      - classify_detections() nhận list bbox (pixel) để phân loại các hộp từ YOLO
      - batched_predict() batch ảnh
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        use_half: bool = True,
        labels: Optional[List[str]] = None,
        templates_vi: Optional[List[str]] = None,
        templates_en: Optional[List[str]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.use_half = use_half and (self.device.startswith("cuda"))
        if self.use_half:
            self.model = self.model.half()

        self.labels = labels or VIETNAMESE_LABELS
        self.templates_vi = templates_vi or DEFAULT_TEMPLATES_VI
        self.templates_en = templates_en or DEFAULT_TEMPLATES_EN

        self.text_features: Optional[torch.Tensor] = None
        self._build_text_features()

    def _build_text_features(self):
        prompts: List[str] = []
        for name in self.labels:
            for t in self.templates_vi:
                prompts.append(t.format(name))
            for t in self.templates_en:
                prompts.append(t.format(name))

        tokens = self.tokenizer(prompts)
        tokens = tokens.to(self.device)

        with torch.no_grad():
            if self.use_half:
                tokens = tokens  
            txt = self.model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)

        per_class_feats = []
        n_template = len(self.templates_vi) + len(self.templates_en)
        for i in range(len(self.labels)):
            s = i * n_template
            e = s + n_template
            per_class_feats.append(txt[s:e].mean(dim=0))
        self.text_features = torch.stack(per_class_feats, dim=0)  

    def set_labels(
        self,
        labels: List[str],
        templates_vi: Optional[List[str]] = None,
        templates_en: Optional[List[str]] = None,
    ):
        self.labels = labels
        if templates_vi is not None:
            self.templates_vi = templates_vi
        if templates_en is not None:
            self.templates_en = templates_en
        self._build_text_features()

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.use_half:
            img = img.half()
        return img

    @torch.inference_mode()
    def classify_image(
        self, image: Image.Image, topk: int = 3, return_probs: bool = True
    ) -> Dict:
        """
        Trả về lớp dự đoán cho một ảnh (crop).
        """
        assert self.text_features is not None, "Text features chưa sẵn sàng."
        img = self._to_tensor(image)
        img_feat = self.model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # (1, D)

        # cosine sim → logits
        logits = (img_feat @ self.text_features.T).squeeze(0)  # (C,)
        probs = logits.softmax(dim=-1)

        topk = min(topk, len(self.labels))
        vals, inds = probs.topk(topk, dim=-1)
        result = {
            "topk_indices": inds.tolist(),
            "topk_labels": [self.labels[i] for i in inds.tolist()],
            "topk_scores": vals.tolist() if return_probs else None,
            "pred_index": int(inds[0].item()),
            "pred_label": self.labels[int(inds[0].item())],
            "pred_score": float(vals[0].item()) if return_probs else None,
            "logits": logits.float().cpu().numpy(),
        }
        return result

    @torch.inference_mode()
    def batched_predict(
        self, images: Sequence[Image.Image], topk: int = 3
    ) -> List[Dict]:
        if len(images) == 0:
            return []
        batch = torch.cat([self._to_tensor(im) for im in images], dim=0)
        img_feat = self.model.encode_image(batch)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # (N, D)

        logits = img_feat @ self.text_features.T  # (N, C)
        probs = logits.softmax(dim=-1)
        topk = min(topk, len(self.labels))
        vals, inds = probs.topk(topk, dim=-1)

        out: List[Dict] = []
        for i in range(len(images)):
            idxs = inds[i].tolist()
            scs = vals[i].tolist()
            out.append(
                {
                    "topk_indices": idxs,
                    "topk_labels": [self.labels[j] for j in idxs],
                    "topk_scores": scs,
                    "pred_index": int(idxs[0]),
                    "pred_label": self.labels[int(idxs[0])],
                    "pred_score": float(scs[0]),
                    "logits": logits[i].float().cpu().numpy(),
                }
            )
        return out

    def classify_detections(
        self,
        image_bgr: np.ndarray,
        bboxes_xyxy: Sequence[Sequence[float]],
        topk: int = 3,
    ) -> List[Dict]:
        """
        Nhận ảnh BGR (numpy, như từ OpenCV) và danh sách bbox [x1,y1,x2,y2] (pixel).
        Trả về list kết quả VLM theo thứ tự bbox.
        """
        # Chuyển sang PIL & crop
        img_rgb = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
        crops: List[Image.Image] = []
        W, H = img_rgb.size
        for (x1, y1, x2, y2) in bboxes_xyxy:
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(W, int(round(x2)))
            y2 = min(H, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                crops.append(img_rgb)  # fallback
            else:
                crops.append(img_rgb.crop((x1, y1, x2, y2)))
        return self.batched_predict(crops, topk=topk)
