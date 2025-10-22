# evaluation/benchmark.py
from __future__ import annotations
import time
import statistics as stats
from typing import List
from PIL import Image
import torch

from .evaluate import YOLOv12Detector, VLMClassifier

def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timeit(fn, warmup=3, repeat=20):
    # warmup
    for _ in range(warmup):
        fn(); _sync_cuda()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        _sync_cuda()
        times.append((time.perf_counter() - t0)*1000.0)  # ms
    return {
        "mean_ms": stats.mean(times),
        "p50_ms": stats.median(times),
        "p90_ms": stats.quantiles(times, n=10)[8],
        "min_ms": min(times),
        "max_ms": max(times),
        "repeat": repeat
    }

def benchmark(detector: YOLOv12Detector, vlm: VLMClassifier, image_paths: List[str], crop_topk: int = 5):
    imgs = [Image.open(p).convert("RGB") for p in image_paths[:20]]

    # Detector latency
    det_res = timeit(lambda: detector.detect(imgs[0]))
    print("[Latency] YOLO detect (single image):", det_res)

    # VLM latency (trên crop giả định 224x224)
    from PIL import ImageOps
    crop = ImageOps.fit(imgs[0], (224,224))
    vlm_res = timeit(lambda: vlm.classify(crop))
    print("[Latency] VLM classify (single crop):", vlm_res)

    # End-to-end (detect + top-k crop + classify)
    def end2end():
        dets = detector.detect(imgs[0])
        dets = sorted(dets, key=lambda d: d.get("score",0), reverse=True)[:crop_topk]
        for d in dets:
            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
            c = imgs[0].crop((x1,y1,x2,y2))
            vlm.classify(c)
    e2e_res = timeit(end2end)
    print(f"[Latency] End-to-end (1 img, top-{crop_topk} crops):", e2e_res)

    # Throughput trên N ảnh
    def batch_run():
        for img in imgs:
            dets = detector.detect(img)
            for d in dets[:crop_topk]:
                x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                c = img.crop((x1,y1,x2,y2))
                vlm.classify(c)
    thr = timeit(batch_run)
    ips = 1000.0 / thr["mean_ms"] * len(imgs)
    print(f"[Throughput] ≈ {ips:.2f} images/s over {len(imgs)} images")

if __name__ == "__main__":
    # ví dụ
    det = YOLOv12Detector(model_path="weights/yolov12.pt")
    vlm = VLMClassifier(prompt="Classify traffic sign")
    # thay bằng một vài ảnh test của bạn
    benchmark(det, vlm, image_paths=["samples/1.jpg","samples/2.jpg"])
