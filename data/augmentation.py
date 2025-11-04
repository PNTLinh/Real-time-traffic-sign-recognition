import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import os
import glob
# Thư mục chứa file label
label_dir = Path("../datasets/processed/processed_train/labels")

class_counter = Counter()

# Đọc từng file label
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        with open(label_dir / label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    class_counter[cls] += 1

# In ra số lượng từng class
for cls, count in class_counter.items():
    print(f"Class {cls}: {count} objects")

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.bar(class_counter.keys(), class_counter.values(), color='skyblue')
plt.xlabel("Class")
plt.ylabel("Số lượng")
plt.title("Số lượng từng class trong dataset")
plt.xticks(list(class_counter.keys()))

# Lưu biểu đồ ra file
output_path = "class_counts.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Biểu đồ đã được lưu tại: {output_path}")

# Hiển thị biểu đồ
plt.show()



# --- 1. ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (Không đổi) ---
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- 2. HÀM AUGMENTATION (Không đổi) ---

def augment_single_file(image_path, label_path, output_image_dir, output_label_dir, base_name, num_augmentations):
    try:
        print(f"    [Bước 3] Đang đọc ảnh: {image_path}")
        image = cv2.imread(image_path)
        if image is None: 
            print(f"     LỖI: Không thể đọc được file ảnh. Bỏ qua.")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
        
        print(f"    [Bước 4] Bắt đầu tạo {num_augmentations} phiên bản augmented...")
        for i in range(num_augmentations):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            new_image_name = f"{base_name}_aug_class19_{i}.jpg"
            new_label_name = f"{base_name}_aug_class19_{i}.txt"
            
            full_image_path = os.path.join(output_image_dir, new_image_name)
            full_label_path = os.path.join(output_label_dir, new_label_name)

            # Lưu ảnh
            cv2.imwrite(full_image_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

            # Lưu label
            with open(full_label_path, 'w') as f:
                for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                    x_center, y_center, width, height = bbox
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
        
        print(f"    [Bước 5] HOÀN TẤT! Đã tạo và lưu {num_augmentations} file mới cho {base_name}.")

    except Exception as e:
        print(f"    LỖI TRONG QUÁ TRÌNH AUGMENT: {e}")

# --- 3. HÀM CHÍNH ĐỂ TÌM VÀ XỬ LÝ ---
def process_target_class(image_dir, label_dir, target_class_id, num_per_image):
    print("================ SCRIPT BẮT ĐẦU ================")
    print(f"[Bước 1] Quét tất cả file .txt trong thư mục:\n  '{os.path.abspath(label_dir)}'")
    
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    if not label_files:
        print("\n!!! LỖI: Không tìm thấy bất kỳ file label (.txt) nào. Vui lòng kiểm tra lại đường dẫn TRAIN_LABEL_DIR.")
        return
        
    print(f" -> Đã tìm thấy tổng cộng {len(label_files)} file label. Bắt đầu kiểm tra từng file...")
    
    found_count = 0
    for label_path in label_files:
        if "_aug_" in label_path:
            continue
            
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        # print(f"  - Đang kiểm tra file: {base_name}.txt") # Bỏ comment dòng này nếu muốn xem TOÀN BỘ file đang quét
        
        contains_target = False
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip().startswith(str(target_class_id) + ' '):
                    contains_target = True
                    break
        
        if contains_target:
            found_count += 1
            print(f"\n[Bước 2] TÌM THẤY CLASS {target_class_id} trong file '{base_name}.txt'")
            
            # Tìm ảnh tương ứng (thử nhiều định dạng)
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                potential_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path:
                augment_single_file(image_path, label_path, image_dir, label_dir, base_name, num_per_image)
            else:
                print(f"    LỖI: Đã tìm thấy label nhưng KHÔNG tìm thấy ảnh tương ứng trong thư mục '{os.path.abspath(image_dir)}'")
                
    print("\n================ SCRIPT KẾT THÚC ================")
    if found_count == 0:
        print(f"-> Không tìm thấy file nào chứa class {target_class_id} trong suốt quá trình quét.")
    else:
        print(f"-> Đã xử lý tổng cộng {found_count} file chứa class {target_class_id}.")

# --- CẤU HÌNH VÀ CHẠY ---
if __name__ == '__main__':
    TRAIN_IMAGE_DIR = '../datasets/processed/processed_train/images'
    TRAIN_LABEL_DIR = '../datasets/processed/processed_train/labels'
    
    # Lam giau cho class 19 gap 25 lan binh thuong
    TARGET_CLASS = 19
    NUM_AUGMENTATIONS_PER_IMAGE = 25
    
    process_target_class(
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        TARGET_CLASS,
        NUM_AUGMENTATIONS_PER_IMAGE
    )

    # Lam giau cho cac class [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18] gap 3 lan binh thuong
    TARGET_CLASSES = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18]
    NUM_NEW_IMAGES_PER_ORIGINAL = 2 # cái mới được aug
    
    print("================ SCRIPT LÀM GIÀU DỮ LIỆU BẮT ĐẦU ================")
    # Lặp qua từng class trong danh sách và thực hiện augmentation
    for target_class in TARGET_CLASSES:
        process_target_class(
            TRAIN_IMAGE_DIR,
            TRAIN_LABEL_DIR,
            target_class,
            NUM_NEW_IMAGES_PER_ORIGINAL
        )

    # Lam giau cho cac class [1, 13] gap 2 lan binh thuong
    TARGET_CLASSES = [1, 13]
    NUM_NEW_IMAGES_PER_ORIGINAL = 1
    print("================ SCRIPT LÀM GIÀU DỮ LIỆU BẮT ĐẦU ================")
    # Lặp qua từng class trong danh sách và thực hiện augmentation
    for target_class in TARGET_CLASSES:
        process_target_class(
            TRAIN_IMAGE_DIR,
            TRAIN_LABEL_DIR,
            target_class,
            NUM_NEW_IMAGES_PER_ORIGINAL
        )






# Visualize so luong class sau khi augment
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Thư mục chứa file label
label_dir = Path("../datasets/processed/processed_train/labels")

class_counter = Counter()

# Đọc từng file label
for label_file in os.listdir(label_dir):
    if label_file.endswith(".txt"):
        with open(label_dir / label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    class_counter[cls] += 1

# In ra số lượng từng class
for cls, count in class_counter.items():
    print(f"Class {cls}: {count} objects")

# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.bar(class_counter.keys(), class_counter.values(), color='skyblue')
plt.xlabel("Class")
plt.ylabel("Số lượng")
plt.title("Số lượng từng class trong dataset")
plt.xticks(list(class_counter.keys()))

# Lưu biểu đồ ra file
output_path = "dataset_after_aug.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Biểu đồ đã được lưu tại: {output_path}")

# Hiển thị biểu đồ
plt.show()