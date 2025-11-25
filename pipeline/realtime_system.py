from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import yaml
import time
import os

from models.vlm_classifier import VLMClassifier


def load_yolo_names(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return [str(n) for n in names]


def draw_box(img, xyxy, color=(0, 255, 0), text: str = None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.rectangle(img, (x1, y1 - 22), (x1 + max(120, 8 * len(text)), y1), color, -1)
        cv2.putText(
            img, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
        )


def main(
    weights_yolo: str,
    data_yaml: str,
    source: int | str = 0,  # webcam id hoặc đường dẫn video
    vlm_model_name: str = "ViT-B-32",
    vlm_pretrained: str = "laion2b_s34b_b79k",
):
    det = YOLO(weights_yolo)
    class_names_yolo = load_yolo_names(data_yaml)

    class_names_vlm = class_names_yolo

    vlm = VLMClassifier(
        labels=class_names_vlm,
        model_name=vlm_model_name,
        pretrained=vlm_pretrained,
        # cache_path="outputs/text_embeds.pt" # Uncomment if applicable
    )

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Không mở được nguồn video: {source}"

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = det.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            pred_vlm = vlm.classify_detections(
                image_bgr=frame,
                bboxes_xyxy=xyxy.tolist(),
                topk=1
            )

            for i, p in enumerate(pred_vlm):
                label = f"{p['pred_label']} {p['pred_score']:.2f}"
                draw_box(frame, xyxy[i], color=(0, 255, 0), text=label)

        cv2.imshow("YOLO + VLM", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    WEIGHTS = "local_root/weights/yolo/best.pt"
    DATA_YAML = "local_root/datasets/data.yaml"
    SOURCE = 0

    main(WEIGHTS, DATA_YAML, SOURCE)
