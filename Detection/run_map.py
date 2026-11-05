import os
import shutil
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')


EVAL_ROOT = "/data/test/biaoqian"
IMAGES_VAL_DIR = os.path.join(EVAL_ROOT, "images/val")

METHODS_IMAGE_DIRS = {
    "Original_Low": "/home/leng/data/qtacefz/data/BUCHONG/Car",
    "CIDNet": "/data/test/data/darkface_exdark/CIDNet/results/Car",
    "DarkIR": "/data/test/data/darkface_exdark/DarkIR/results/Car",
    "DMFourLLIE": "/data/test/data/darkface_exdark/DMFourLLIE/results/Car",
    "sfhformer": "/data/test/data/darkface_exdark/sfhformer/results/Car",
    "ou": "/data/test/data/darkface_exdark/ou/results/Car",
    "RetinexNet": "/data/test/data/darkface_exdark/RetinexNet/results/Car",
    "EnlightenGAN": "/data/test/data/darkface_exdark/EnlightenGAN/results/Car",
    "FourLLIE": "/data/test/data/darkface_exdark/FourLLIE/results/Car",
    "RUAS": "/data/test/data/darkface_exdark/RUAS/results/Car",
    "LLFormer": "/data/test/data/darkface_exdark/LLFormer/results/Car",
}

print("...")
model = YOLO("yolov8n.pt").to("cuda:0")

def copy_images_for_testing(src_dir, target_dir):

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for f in os.listdir(src_dir):
        if f.endswith(('.jpg', '.png', '.jpeg')):

            prefix = f[:10]
            ext = os.path.splitext(f)[1]
            new_name = prefix + ext
            shutil.copy(os.path.join(src_dir, f), os.path.join(target_dir, new_name))


results = {}

for method_name, img_dir in METHODS_IMAGE_DIRS.items():
    print(f"\n{'=' * 55}")
    print(f" {method_name}")
    print(f"{'=' * 55}")

    if not os.path.exists(img_dir):
        print(f"{img_dir}")
        results[method_name] = {"mAP50": 0.0, "mAP50-95": 0.0}
        continue

    copy_images_for_testing(img_dir, IMAGES_VAL_DIR)

    metrics = model.val(
        data=os.path.join(EVAL_ROOT, "dataset.yaml"),
        conf=0.001,
        iou=0.6,
        split='val',
        verbose=False
    )

    try:
        class_idx = metrics.box.ap_class_index.tolist().index(2)
        mAP_50 = metrics.box.ap50[class_idx]
        mAP_50_95 = metrics.box.ap[class_idx]
    except ValueError:
        mAP_50 = 0.0
        mAP_50_95 = 0.0

    results[method_name] = {"mAP50": mAP_50, "mAP50-95": mAP_50_95}
    print(f"mAP@50: {mAP_50 * 100:.2f}%")

print("\n\n" + "*" * 55)

print("*" * 55)
print(f"{'Method':<20} | {'mAP@50 (%)':<12} | {'mAP@50-95 (%)':<12}")
print("-" * 55)
for method, res in results.items():
    print(f"{method:<20} | {res['mAP50'] * 100:>8.2f}     | {res['mAP50-95'] * 100:>8.2f}")
print("*" * 55)