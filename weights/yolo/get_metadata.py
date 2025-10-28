import torch
from ultralytics import YOLO
import pprint # Dùng để in dictionary cho đẹp
import json # <-- Thêm module json
from pathlib import Path # <-- Thêm module pathlib để xử lý đường dẫn

# --- ĐƯỜNG DẪN ĐẾN MODEL CỦA BẠN ---
# Lấy từ file bạn cung cấp
WEIGHTS = r"C:\Users\Admin\OneDrive\Desktop\deeplearning_project\Real-time-traffic-sign-recognition\weights\yolo\best.pt"

# --- ĐƯỜNG DẪN FILE JSON ĐẦU RA ---
JSON_OUTPUT_PATH = "./model_in4.json"

print(f"🔍 Đang tải metadata từ: {WEIGHTS}\n")

# Tạo một dictionary để chứa tất cả thông tin
metadata_dict = {}

try:
    # 1. Tải mô hình bằng class YOLO
    # Thao tác này sẽ tự động đọc metadata
    model = YOLO(WEIGHTS)

    # 2. Lấy dictionary checkpoint gốc
    # (Đây là dữ liệu thô được lưu bên trong file .pt)
    ckpt = model.ckpt 

    metadata_dict['model_path'] = WEIGHTS

    print("="*30)
    print("    THÔNG TIN CƠ BẢN")
    print("="*30)
    
    # In tên các lớp
    print("\n--- 🏷️ Tên các lớp (model.names) ---")
    pprint.pprint(model.names)
    # Thêm vào dict
    metadata_dict['class_names'] = model.names

    # In epoch
    if 'epoch' in ckpt:
        epoch = ckpt['epoch']
        print(f"\n---  EPOCH ---")
        print(f"Model này được lưu ở epoch: {epoch}")
        # Thêm vào dict
        metadata_dict['epoch'] = epoch
    else:
        metadata_dict['epoch'] = None # Ghi là None nếu không tìm thấy

    # In kết quả tốt nhất (thường là mAP50-95)
    if 'best_fitness' in ckpt:
        fitness = ckpt['best_fitness']
        print(f"\n--- 📈 Kết quả (Best Fitness) ---")
        
        # --- SỬA LỖI: Kiểm tra xem fitness có phải là None không trước khi format ---
        if fitness is not None:
            print(f"Chỉ số tốt nhất (thường là mAP50-95): {fitness:.4f}")
            metadata_dict['best_fitness'] = fitness
        else:
            print("Chỉ số 'best_fitness' là None (model có thể chưa training).")
            metadata_dict['best_fitness'] = None
        # ---------------------------------------------------------------------

    else:
        metadata_dict['best_fitness'] = None # Ghi là None nếu không tìm thấy

    print("\n" + "="*30)
    print(" CẤU HÌNH LÚC TRAINING (ckpt['train_args'])") # <-- Lấy từ ckpt
    print("="*30)
    
    # --- SỬA LỖI: Lấy 'train_args' từ ckpt, không dùng model.args ---
    # model.args gây lỗi "vars()", 'train_args' (từ log) đã là một dict
    if 'train_args' in ckpt and ckpt['train_args'] is not None:
        
        # Nó đã là một dictionary, không cần vars()
        args_dict = ckpt['train_args'] 
        
        pprint.pprint(args_dict) # Vẫn in ra console

        # Chuyển đổi args sang dạng có thể lưu JSON
        # (Xử lý các đối tượng Path thành string)
        serializable_args = {}
        for k, v in args_dict.items(): # <-- Dùng args_dict.items()
            if isinstance(v, Path):
                serializable_args[k] = str(v) # Chuyển Path thành string
            # Chỉ lấy các kiểu dữ liệu cơ bản mà JSON hỗ trợ
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_args[k] = v
            else:
                # Nếu là kiểu dữ liệu lạ (ví dụ: một object), chỉ lưu dạng string
                serializable_args[k] = str(v) 
        
        # Thêm vào dict
        metadata_dict['training_args'] = serializable_args
    else:
        print("Không tìm thấy thông tin 'train_args' trong checkpoint.")
        metadata_dict['training_args'] = None
    # -------------------------------------------------------------------

    # 3. Lưu tất cả thông tin ra file JSON
    print("\n" + "="*30)
    print(f"💾 Đang lưu metadata vào file {JSON_OUTPUT_PATH}...")
    
    with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để lưu đúng tiếng Việt
        # indent=4 để file JSON dễ đọc
        json.dump(metadata_dict, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Đã lưu thành công!")


except Exception as e:
    print(f"❌ Xảy ra lỗi khi đọc file: {e}")
    print("\n--- THỬ ĐỌC BẰNG TORCH (CÁCH DỰ PHÒNG) ---")
    try:
        # Cách dự phòng: Tải bằng torch
        
        # --- SỬA LỖI: Thêm weights_only=False cho PyTorch 2.6+ ---
        # Chỉ làm điều này nếu bạn tin tưởng file checkpoint
        print("Cảnh báo: Tải file với 'weights_only=False' (do lỗi bảo mật của PyTorch 2.6+).")
        raw_ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False) # map_location='cpu' để không yêu cầu GPU
        # -------------------------------------------------------
        
        print("Đã tải file checkpoint thô. Các 'key' có trong file:")
        pprint.pprint(raw_ckpt.keys())
        
        if 'train_args' in raw_ckpt: # <-- Đổi 'args' thành 'train_args'
            print("\n--- Cấu hình 'train_args' (thô) ---")
            pprint.pprint(raw_ckpt['train_args'])
        
        if 'model' in raw_ckpt:
             print("\nModel state_dict có " 
                   f"{len(raw_ckpt['model'].state_dict().keys())} lớp (layers).")

    except Exception as e_torch:
        print(f"❌ Lỗi khi tải bằng torch: {e_torch}")

