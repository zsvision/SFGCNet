import os
import cv2
import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


LOW_DIR = "/root/our/data/BUCHONG/track1.2_test_sample"  

METHODS = {
    "CIDNet":      ("/root/our/darkface_exdark/CIDNet/results/track1.2_test_sample", "_cidnet"),
    "DarkIR":      ("/root/our/darkface_exdark/DarkIR/results/track1.2_test_sample", "_darkir"),
    "DMFourLLIE":  ("/root/our/darkface_exdark/DMFourLLIE/results/track1.2_test_sample", "_dmfourllie"),
    "sfhformer":   ("/root/our/darkface_exdark/sfhformer/results/track1.2_test_sample", "_sfhformer"),
    "ou":          ("/root/our/darkface_exdark/ou/results/track1.2_test_sample", "_ou"),
    "RetinexNet":  ("/root/our/darkface_exdark/RetinexNet/results/track1.2_test_sample", "_RetinexNet"),
    "EnlightenGAN":("/root/our/darkface_exdark/EnlightenGAN/results/track1.2_test_sample", "_EnlightenGAN"),
    "FourLLIE":    ("/root/our/darkface_exdark/FourLLIE/results/track1.2_test_sample", "_FourLLIE"),
    "RUAS":  ("/root/our/darkface_exdark/RUAS/results/track1.2_test_sample", "_RUAS"),
    "LLFormer":  ("/root/our/darkface_exdark/LLFormer/results/track1.2_test_sample", "_LLFormer"),
}

SAVE_ROOT = "./detection_track1.2_test_sample"
CONF_THRESH = 0.3
# ========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}")
model = YOLO("./yolov8n.pt").to(device)


save_low = os.path.join(SAVE_ROOT, "track1.2_test_sample_low")
os.makedirs(save_low, exist_ok=True)


img_names = sorted([f for f in os.listdir(LOW_DIR) if f.endswith(('jpg', 'png'))])
for name in img_names:
    img = cv2.imread(os.path.join(LOW_DIR, name))
    if img is None: continue
    det_img = model(img, conf=CONF_THRESH, verbose=False)[0].plot()
    cv2.imwrite(os.path.join(save_low, name), det_img)

for method_name, (enh_dir, suffix) in METHODS.items():
    save_dir = os.path.join(SAVE_ROOT, method_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"{method_name}")

    enh_map = {}
    for f in os.listdir(enh_dir):
        if not f.endswith(('jpg', 'png')):
            continue

        prefix = f.split('_')[0]
        enh_map[prefix] = f

    for name in img_names:

        prefix = os.path.splitext(name)[0]
        if prefix not in enh_map:
            print(f" {name} ")
            continue

        enh_path = os.path.join(enh_dir, enh_map[prefix])
        enh_img = cv2.imread(enh_path)
        if enh_img is None:
            continue

        det_img = model(enh_img, conf=CONF_THRESH, verbose=False)[0].plot()

        save_name = f"{prefix}{suffix}.jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), det_img)

print("\n all")