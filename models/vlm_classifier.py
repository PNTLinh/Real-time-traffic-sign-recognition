import torch
import clip
from PIL import Image
import numpy as np
import os
import cv2

class VLMClassifier:
    def __init__(self, cache_path="weights/vlm/text_features_v16.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_path = cache_path
        
        self.class_names = [
            "Cam Do Xe", "Cam Dung Do Xe", "Cam Nguoc Chieu", "Cam Oto", 
            "Cam Quay Dau", "Cam Re Phai", "Cam Re Trai", "Dung Lai", 
            "Duong Khong Bang Phang", "Duong Khong Uu Tien", "Duong Uu Tien", 
            "Nguoi Bi Bo", "Toc Do 30", "Toc Do 40", "Toc Do 50", 
            "Toc Do 60", "Toc Do 80", "Tre Em Qua Duong", "Vong Xuyen"
        ]

        self.english_prompts = {
            "Cam Do Xe": "No parking sign",
            "Cam Dung Do Xe": "No stopping and parking sign",
            "Cam Nguoc Chieu": "Do not enter sign",
            "Cam Oto": "No cars allowed sign",
            "Cam Quay Dau": "No U-turn sign",
            "Cam Re Phai": "No right turn sign",
            "Cam Re Trai": "No left turn sign",
            "Dung Lai": "Stop sign",
            "Duong Khong Bang Phang": "Bumpy road warning sign",
            "Duong Khong Uu Tien": "Yield sign", 
            "Duong Uu Tien": "Priority road sign",
            "Nguoi Bi Bo": "Pedestrian crossing sign",
            "Toc Do 30": "Speed limit 30 km/h sign",
            "Toc Do 40": "Speed limit 40 km/h sign",
            "Toc Do 50": "Speed limit 50 km/h sign",
            "Toc Do 60": "Speed limit 60 km/h sign",
            "Toc Do 80": "Speed limit 80 km/h sign",
            "Tre Em Qua Duong": "Children crossing warning sign",
            "Vong Xuyen": "Roundabout ahead sign"
        }
        
        print(f"Loading OpenCLIP model: ViT-B/16 ...")
        try:
            self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        except Exception:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        if os.path.exists(cache_path):
            print("Checking cache compatibility...")
            try:
                tmp = torch.load(cache_path, map_location=self.device, weights_only=False)
                if tmp.shape[1] != self.model.text_projection.shape[1]:
                    print("Dimension mismatch. Regenerating cache...")
                    os.remove(cache_path)
            except:
                pass

        self.load_or_generate_text_features()

        print(f"VLM Classifier ready on {self.device}")

    def load_or_generate_text_features(self):
        if os.path.exists(self.cache_path):
            print("Loading cached text features ...")
            try:
                self.text_features = torch.load(self.cache_path, map_location=self.device, weights_only=False)
                return
            except Exception as e:
                print(f"Cache error ({e}). Regenerating...")

        print("Generating new text features (English)...")
        prompts = [f"A photo of a {self.english_prompts[c]}" for c in self.class_names]
        
        text_inputs = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        self.text_features = text_features
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save(self.text_features, self.cache_path)
        print(f"Saved features to {self.cache_path}")

    def classify_crops(self, frame, boxes):
        if not boxes: return []
        pil_images = []
        valid_boxes = [] 
        h_img, w_img = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            pad = int(max(x2-x1, y2-y1) * 0.15) + 5 
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w_img, x2 + pad)
            y2 = min(h_img, y2 + pad)

            if x2 <= x1 or y2 <= y1: continue 

            crop = frame[y1:y2, x1:x2]
            
            if crop.shape[0] < 100 or crop.shape[1] < 100:
                 crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(crop_rgb))
            valid_boxes.append(box)

        if not pil_images: return []

        try:
            image_input = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return []

        with torch.no_grad():
            with torch.amp.autocast('cuda' if 'cuda' in str(self.device) else 'cpu'):
                img_feat = self.model.encode_image(image_input)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                
                if self.text_features.dtype != img_feat.dtype:
                    self.text_features = self.text_features.to(img_feat.dtype)
                
                logits = 100.0 * img_feat @ self.text_features.T
                probs = logits.softmax(dim=-1)

        results = []
        vals, ids = probs.topk(3)
        
        for i in range(len(valid_boxes)):
            idx = int(ids[i][0])
            score = float(vals[i][0])
            label = self.class_names[idx]
            
            print(f"Box {valid_boxes[i]}:")
            print(f"   1. {label}: {score:.4f}")
            print(f"   2. {self.class_names[int(ids[i][1])]}: {float(vals[i][1]):.4f}")
            print(f"   3. {self.class_names[int(ids[i][2])]}: {float(vals[i][2]):.4f}")

            results.append({
                "box": valid_boxes[i],
                "pred_label": label,
                "pred_score": score
            })
            
        return results