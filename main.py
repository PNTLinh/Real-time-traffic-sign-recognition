import argparse
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from pipeline.realtime_system import RealTimeSystem
from models.yolo_detector import YOLODetector
from models.yolo_trainer import train_yolo
from pipeline.optimizer import ModelOptimizer
from utils.logger import setup_logger
from ultralytics import YOLO

def load_config(path="config.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Cannot load config {path}: {e}")
        sys.exit(1)

def train_mode(args):
    print("\nTRAINING MODE\n" + "=" * 60)

    model, results = train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    print("\nTraining completed!")

def inference_mode(args, config):
    print("\nINFERENCE MODE\n" + "=" * 60)

    enable_vlm = not args.no_vlm

    system = RealTimeSystem(
        model_path=args.model,
        camera_id=0,
        img_size=config.get("img_size", 320),
        show_vlm=enable_vlm,
        batch_vlm=True,
    )

    if args.source == "video":
        if args.input is None:
            print("You must specify --input for video mode")
            sys.exit(1)
        print(f"Running on Video File: {args.input}")
        system.run_video(args.input)

    elif args.source == "image":
        if args.input is None:
            print("You must specify --input for image mode")
            sys.exit(1)
        print(f"Processing Image: {args.input}")
        system.run_image(args.input)

    else:
        print(f"Unknown source: {args.source}")
        sys.exit(1)

def optimize_mode(args):
    print("\nOPTIMIZATION MODE\n" + "=" * 60)

    opt = ModelOptimizer()

    if args.optimize_action == "onnx":
        print("Exporting to ONNX...")
        YOLO(model=args.model).export(format="onnx")

    elif args.optimize_action == "tensorrt":
        print("Exporting to TensorRT...")
        opt.export_to_trt(
            model_path=args.model,
            output=args.output or "weights/yolo/best.engine",
        )

    elif args.optimize_action == "benchmark":
        print("Benchmarking ...")
        opt.benchmark(
            model_path=args.model,
            imgsz=args.imgsz,
            num_iterations=args.iterations,
        )

    print("\nOptimization completed!")

def test_mode(args):
    print("\nTEST MODE\n" + "=" * 60)

    detector = YOLODetector(args.model, conf=0.2, iou=0.45)

    import cv2
    if args.input is None:
        print(" You must specify --input for test mode")
        return

    image = cv2.imread(args.input)
    if image is None:
        print(f"Cannot load image: {args.input}")
        return

    dets = detector.detect(image)

    print(f"Found {len(dets)} detections:")
    for d in dets:
        print(f" - {d['class_name']} ({d['confidence']:.2f})")

    vis = detector.visualize(image, dets)
    out = "outputs/test_result.jpg"
    cv2.imwrite(out, vis)
    print(f"Saved: {out}")

    print("\nTest completed!")

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Traffic Sign Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--mode", required=True,
                        choices=["train", "inference", "optimize", "test"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="weights/yolo/best.pt")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)

    # Training
    parser.add_argument("--data", default="datasets/processed/data.yaml")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--imgsz", default=320, type=int)

    # Inference
    parser.add_argument("--source", choices=["video", "image"])
    parser.add_argument("--no-vlm", action="store_true")

    # Optimization
    parser.add_argument("--optimize-action",
                        choices=["onnx", "tensorrt", "quantize", "benchmark"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    # Logging
    setup_logger()

    # Load config
    config = load_config(args.config)

    print("\n" + "=" * 60)
    print("TRAFFIC SIGN RECOGNITION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print("=" * 60 + "\n")

    if args.mode == "train":
        train_mode(args)

    elif args.mode == "inference":
        inference_mode(args, config)

    elif args.mode == "optimize":
        if args.optimize_action is None:
            print("Please provide --optimize-action")
            sys.exit(1)
        optimize_mode(args)

    elif args.mode == "test":
        test_mode(args)

    print("\n" + "=" * 60)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
