import os
import csv
CSV_PATH = "/data/test/biaoqian/annotations/annotations.csv"

IMG_DIR = "/home/leng/data/qtacefz/data/BUCHONG/Car"

YOLO_LABEL_DIR = "/data/test/biaoqian/labels/val"
# =======================================================

os.makedirs(YOLO_LABEL_DIR, exist_ok=True)
count = 0
yolo_data = {}

print("...")

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
    
        full_name = os.path.basename(row['image_path'])
        base_name = os.path.splitext(full_name)[0]

    
        cls_str = str(row['class']).strip()
        if cls_str != '4':
            continue 

        try:
            x_tl = float(row['x_tl'])
            y_tl = float(row['y_tl'])
            w_box = float(row['w'])
            h_box = float(row['h'])
            img_width = float(row['image_width'])
            img_height = float(row['image_height'])
        except (ValueError, KeyError):
            continue

        x_center = max(0.0, min(1.0, (x_tl + w_box / 2.0) / img_width))
        y_center = max(0.0, min(1.0, (y_tl + h_box / 2.0) / img_height))
        w_norm = max(0.0, min(1.0, w_box / img_width))
        h_norm = max(0.0, min(1.0, h_box / img_height))

  
        class_id = 2
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        if base_name not in yolo_data:
            yolo_data[base_name] = []
        yolo_data[base_name].append(line)


img_names_in_dir = [os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]

for base_name in img_names_in_dir:
    if base_name in yolo_data:
        yolo_txt_path = os.path.join(YOLO_LABEL_DIR, base_name + ".txt")
        with open(yolo_txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_data[base_name]))
        count += 1

print(f" {YOLO_LABEL_DIR}")