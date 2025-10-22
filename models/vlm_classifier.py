import os
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip  
import yaml

class VLMClassifier:
    """
    Phân loại crop (biển báo) bằng CLIP/OpenCLIP.
    - Tiền xử lý text: tạo embedding cho mỗi lớp từ template prompt.
    - Inference: encode crop -> so khớp cosine với text embeddings.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.class_names = class_names or []
        self.templates = templates or [
            "a photo of a traffic sign: {}",
            "a road sign that indicates {}",
            "a pictogram of {} traffic sign",
            "a sign saying {}",
        ]

        self.cache_path = cache_path  # file .pt để cache text embeddings
        self.text_features = None  # Tensor [C, D], đã L2-normalize

        if self.class_names:
            self.build_text_features(self.class_names, self.templates)

    @staticmethod
    def load_names_from_yaml(yaml_path: str) -> List[str]:
        """
        Đọc danh sách lớp từ datasets/data.yaml (YOLO format).
        Trả về list[str] theo thứ tự id.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        names = y.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        assert isinstance(names, list),"names trong YAML phải là list hoặc dict"
        return [str(n) for n in names]

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    def _encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            feats = self.model.encode_text(tokens.to(self.device))
        return self._normalize(feats.float())

    def build_text_features(self, class_names: List[str], templates: Optional[List[str]] = None) -> None:
        """
        Tạo embedding văn bản cho từng lớp bằng nhiều template rồi average.
        Lưu cache nếu có cache_path.
        """
        templates = templates or self.templates
        cache_ok = False

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                data = torch.load(self.cache_path, map_location=self.device)
                if isinstance(data, dict) and data.get("class_names") == class_names:
                    self.text_features = data["text_features"].to(self.device)
                    cache_ok = True
            except Exception:
                pass

        if not cache_ok:
            all_feats = []
            for cls in class_names:
                prompts = [t.format(cls) for t in templates]
                feats = self._encode_text_prompts(prompts)  # [T, D]
                feats = self._normalize(feats.mean(dim=0, keepdim=True))  # [1, D]
                all_feats.append(feats)
            self.text_features = torch.cat(all_feats, dim=0)  # [C, D]

            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                torch.save({"class_names": class_names, "text_features": self.text_features.cpu()}, self.cache_path)

        self.class_names = class_names

    # ---------- Inference ----------
    def _encode_image(self, img_pil: Image.Image) -> torch.Tensor:
        img = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            feats = self.model.encode_image(img)
        return self._normalize(feats.float())  # [1, D]

    def predict_crops(
        self,
        image: Image.Image,
        boxes_xyxy: List[Tuple[float, float, float, float]],
        expand: float = 0.08,
        topk: int = 1,
    ) -> List[Dict]:
        """
        Phân loại cho danh sách crop từ ảnh gốc.
        boxes_xyxy: [(x1,y1,x2,y2), ...] theo pixel.
        expand: nới rộng box (tỷ lệ cạnh) để lấy đủ bối cảnh.
        Trả về: list dict {bbox, cls_id, cls_name, score}
        """
        assert self.text_features is not None, "Chưa build_text_features (chưa có class_names)"
        w, h = image.size
        results = []

        for (x1, y1, x2, y2) in boxes_xyxy:
            # nới box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = (x2 - x1), (y2 - y1)
            bw, bh = bw * (1 + expand), bh * (1 + expand)
            nx1, ny1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            nx2, ny2 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))

            crop = image.crop((nx1, ny1, nx2, ny2))
            img_feat = self._encode_image(crop)  # [1, D]

            # Cosine similarity với text features
            logits = (img_feat @ self.text_features.t()).squeeze(0)  # [C]
            probs = logits.softmax(dim=-1)

            if topk == 1:
                score, cls_id = probs.max(dim=-1)
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": int(cls_id.item()),
                    "cls_name": self.class_names[int(cls_id)],
                    "score": float(score.item())
                })
            else:
                score, cls_id = probs.topk(topk, dim=-1)
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "topk": [
                        {"cls_id": int(i), "cls_name": self.class_names[int(i)], "score": float(s)}
                        for s, i in zip(score.tolist(), cls_id.tolist())
                    ]
                })

        return results
