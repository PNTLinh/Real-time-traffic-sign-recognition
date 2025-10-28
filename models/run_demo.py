import argparse, cv2, yaml, os, time
from pathlib import Path
from typing import List
from ultralytics import YOLO
from PIL import Image

from models.vlm_classifier import VLMClassifier

def load_names(yaml_path: str) -> List[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x:int(x))]
    return [str(n) for n in names]

def draw_box(im, x1, y1, x2, y2, text, color=(0,255,0)):
    cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
    (tw,th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(th+6, y1-6)
    cv2.rectangle(im, (x1, y-th-6), (x1+tw+6, y+bl), color, -1)
    cv2.putText(im, text, (x1+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def run_image(model, vlm, img_path: Path, out_path: Path, conf: float, imgsz: int, device: str, expand: float):
    r = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    im = r.orig_img.copy()
    names = r.names
    xyxy = [tuple(map(float, b.xyxy[0].tolist())) for b in r.boxes]
    cls_ids = [int(b.cls[0].item()) for b in r.boxes]
    confs   = [float(b.conf[0].item()) for b in r.boxes]

    vlm_txt = [None]*len(xyxy)
    if vlm and len(xyxy):
        im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        preds = vlm.predict_crops(im_pil, xyxy, expand=expand, topk=1)
        vlm_txt = [f"{p['cls_name']} {p['score']:.2f}" for p in preds]

    for (x1,y1,x2,y2), cid, cf, vt in zip(xyxy, cls_ids, confs, vlm_txt):
        t = f"{names[cid]} {cf:.2f}" + (f" | {vt}" if vt else "")
        draw_box(im, int(x1), int(y1), int(x2), int(y2), t)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), im)
    print("Saved:", out_path)

def run_video(model, vlm, src, out_path: Path, conf: float, imgsz: int, device: str, expand: float):
    cap = cv2.VideoCapture(src)
    assert cap.isOpened(), f"Không mở được video/webcam: {src}"
    w,h,fps = int(cap.get(3)), int(cap.get(4)), cap.get(5) or 25.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    t0 = time.time(); n=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        r = model.predict(source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        im = r.orig_img.copy()
        names = r.names
        xyxy = [tuple(map(float, b.xyxy[0].tolist())) for b in r.boxes]
        cls_ids = [int(b.cls[0].item()) for b in r.boxes]
        confs   = [float(b.conf[0].item()) for b in r.boxes]

        vlm_txt = [None]*len(xyxy)
        if vlm and len(xyxy):
            im_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            preds = vlm.predict_crops(im_pil, xyxy, expand=expand, topk=1)
            vlm_txt = [f"{p['cls_name']} {p['score']:.2f}" for p in preds]

        for (x1,y1,x2,y2), cid, cf, vt in zip(xyxy, cls_ids, confs, vlm_txt):
            t = f"{names[cid]} {cf:.2f}" + (f" | {vt}" if vt else "")
            draw_box(im, int(x1), int(y1), int(x2), int(y2), t)

        vw.write(im); n+=1
    vw.release(); cap.release()
    print(f"Saved video: {out_path} | frames={n} | {time.time()-t0:.1f}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo/best.pt", help="YOLO .pt (hoặc yolo11n.pt/yolov8n.pt)")
    ap.add_argument("--data", default="datasets/data.yaml", help="YOLO data.yaml để lấy tên lớp")
    ap.add_argument("--source", required=True, help="ảnh / thư mục / video / webcam id (0)")
    ap.add_argument("--save-dir", default="outputs/pipeline")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")  # 'cpu' hoặc '0'
    ap.add_argument("--use-vlm", action="store_true")
    ap.add_argument("--vlm-model", default="ViT-B-32")
    ap.add_argument("--vlm-pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--vlm-expand", type=float, default=0.08)
    args = ap.parse_args()

    # load YOLO
    try:
        model = YOLO(args.weights)
    except Exception:
        print(f"[!] Không load được {args.weights}, fallback sang yolo11n.pt")
        model = YOLO("C:\Users\ntlinh\Documents\20251\DL\Real-time-traffic-sign-recognition\weights\yolo\best.pt") 

    names = load_names(args.data)
    vlm = None
    if args.use_vlm:
        vlm = VLMClassifier(
            class_names=names,
            model_name=args.vlm_model,
            pretrained=args.vlm_pretrained,
            device=None,
            cache_path="cache/text_feats.pt"
        )

    save_dir = Path(args.save_dir)
    src = args.source
    if src.isdigit():  # webcam
        run_video(model, vlm, int(src), save_dir/"webcam_pred.mp4", args.conf, args.imgsz, args.device, args.vlm_expand)
    else:
        p = Path(src)
        if p.is_dir():
            exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
            for im_path in sorted(x for x in p.rglob("*") if x.suffix.lower() in exts):
                out_path = save_dir / f"{im_path.stem}_pred.jpg"
                run_image(model, vlm, im_path, out_path, args.conf, args.imgsz, args.device, args.vlm_expand)
        elif p.suffix.lower() in {".mp4",".avi",".mov",".mkv"}:
            out_path = save_dir / f"{p.stem}_pred.mp4"
            run_video(model, vlm, str(p), out_path, args.conf, args.imgsz, args.device, args.vlm_expand)
        else:  # ảnh đơn
            out_path = save_dir / f"{p.stem}_pred.jpg"
            run_image(model, vlm, p, out_path, args.conf, args.imgsz, args.device, args.vlm_expand)

if __name__ == "__main__":
    main()
