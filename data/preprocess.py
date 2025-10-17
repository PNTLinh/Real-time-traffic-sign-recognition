# Load và preprocess dataset

import os
import cv2
from pathlib import Path
from collections import Counter


# Thu muc chua anh goc
original_dir = Path("../datasets/raw/raw_train/images")

# Định dạng ảnh phổ biến
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif"}

sizes = []

for filename in os.listdir(original_dir):
    if any(filename.lower().endswith(ext) for ext in IMG_EXTS):
        img_path = original_dir / filename
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            sizes.append((w, h))


# Dem so lan xuat hien cua moi kich thuoc
size_counts = Counter(sizes)
for size, count in size_counts.items():
    print(f"Kích thước: {size[0]}x{size[1]} - Số lượng: {count}")




# Thu muc dau vao cho viec tien xu ly
input_img_dir = Path("../datasets/raw/raw_train/images")
input_lbl_dir = Path("../datasets/raw/raw_train/labels")
output_img_dir = Path("../datasets/processed/processed_train/images")
output_lbl_dir = Path("../datasets/processed/processed_train/labels")

# File valid
# input_img_dir = Path("../datasets/raw/raw_valid/images")
# input_lbl_dir = Path("../datasets/raw/raw_valid/labels")
# output_img_dir = Path("../datasets/processed/processed_valid/images")
# output_lbl_dir = Path("../datasets/processed/processed_valid/labels")

# File test
# input_img_dir = Path("../datasets/raw/raw_test/images")
# input_lbl_dir = Path("../datasets/raw/raw_test/labels")
# output_img_dir = Path("../datasets/processed/processed_test/images")
# output_lbl_dir = Path("../datasets/processed/processed_test/labels")
# Kích thước mục tiêu để thay đổi kích thước ảnh
target_size = (320, 320)

# Định dạng ảnh phổ biến
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif"}

# Tạo thư mục đầu ra nếu chưa tồn tại
output_img_dir.mkdir(parents=True, exist_ok=True)
output_lbl_dir.mkdir(parents=True, exist_ok=True)

# Tạo thư mục đầu vao nếu chưa tồn tại
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)
# Doc anh tu nhieu dinh dang
def read_image_any_format(image_path):
    image_path = Path(image_path)
    ext = image_path.suffix.lower()
    if ext in IMG_EXTS:
        img = cv2.imread(str(image_path))
        return img
    else:
        raise ValueError(f"Định dạng ảnh không được hỗ trợ: {ext}")

# Ham de them vien den va thay doi kich thuoc anh
def letterbox_image(img, labels, target_size):
    h, w = img.shape[:2]
    if isinstance(target_size, (tuple, list)):
        target_size = target_size[0]  
    scale = min(target_size / w, target_size / h)
    new_h, new_w = int(h * scale), int(w * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    padded_img = cv2.copyMakeBorder(resized_img, pad_y, target_size - new_h - pad_y,
                                    pad_x, target_size - new_w - pad_x,
                                    cv2.BORDER_CONSTANT, value=[114, 114, 114]) # Mau nen xam chuan cho yolo
    
    new_labels = []
    for label in labels:
        cls, x, y, w_box, h_box = label
        x *= w; y *= h; w_box *= w; h_box *= h
        x = x * scale + pad_x
        y = y * scale + pad_y
        w_box *= scale
        h_box *= scale
        x /= target_size
        y /= target_size
        w_box /= target_size
        h_box /= target_size
        new_labels.append((cls, x, y, w_box, h_box))

    return padded_img, new_labels



for filename in os.listdir(input_img_dir):
    ext = Path(filename).suffix.lower()
    if ext not in IMG_EXTS:
        continue

    name = Path(filename).stem
    img_path = os.path.join(input_img_dir, filename)
    label_path = os.path.join(input_lbl_dir, f"{name}.txt")

    image = read_image_any_format(img_path)
    if image is None:
        print(f"Bỏ qua {filename}")
        continue

    # đọc label
    labels = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(list(map(float, parts)))

    padded, new_labels = letterbox_image(image, labels, target_size)

    # lưu ảnh (dạng .jpg)
    cv2.imwrite(os.path.join(output_img_dir, f"{name}.jpg"), padded)

    # lưu label
    with open(os.path.join(output_lbl_dir, f"{name}.txt"), "w") as f:
        for cls, x, y, w_box, h_box in new_labels:
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

    print(f"Xử lý {filename}")

print("\n Hoàn tất resize + pad tất cả ảnh về 320×320!")



# Kiem tra anh sau khi resize + pad
import matplotlib.pyplot as plt

def show_image_with_labels(img, labels):
    img_copy = img.copy()
    h, w = img.shape[:2]

    for label in labels:
        cls, x, y, bw, bh = label
        # chuyển từ YOLO format (x,y,w,h chuẩn hóa) sang pixel
        cx, cy = int(x * w), int(y * h)
        box_w, box_h = int(bw * w), int(bh * h)

        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, f"{int(cls)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


import os
import random
from pathlib import Path

def show_image_with_labels(img, labels, title=""):
    img_copy = img.copy()
    h, w = img.shape[:2]

    for label in labels:
        cls, x, y, bw, bh = label
        # chuyển YOLO (x,y,w,h) → pixel
        cx, cy = int(x * w), int(y * h)
        box_w, box_h = int(bw * w), int(bh * h)
        x1, y1 = int(cx - box_w / 2), int(cy - box_h / 2)
        x2, y2 = int(cx + box_w / 2), int(cy + box_h / 2)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, str(int(cls)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# =========================
# 💡 Hiển thị ngẫu nhiên vài ảnh
# =========================
num_samples = 5   # số ảnh muốn xem
sample_files = random.sample(os.listdir(output_img_dir), num_samples)

for file in sample_files:
    img_path = os.path.join(output_img_dir, file)
    lbl_path = os.path.join(output_lbl_dir, Path(file).stem + ".txt")

    img = cv2.imread(img_path)
    labels = []

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append(list(map(float, parts)))

    show_image_with_labels(img, labels, title=file)
