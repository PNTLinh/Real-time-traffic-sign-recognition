import torch
from ultralytics import YOLO

def train_yolo(
    data_yaml='datasets/processed/data.yaml',
    epochs=100,
    imgsz=320,
    batch=16,
    model_size='n'
):
    """
    Huấn luyện YOLO model
    """
    model = YOLO("yolo11x.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='outputs/yolo',
        name='train',
        save=True,
        pretrained=True,
        verbose=True
    )

    metrics = model.val()
    print("\n📊 Validation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    model.export(format='onnx')
    print("✅ Model exported to ONNX")

    return model, results
