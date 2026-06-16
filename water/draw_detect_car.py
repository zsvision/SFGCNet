import os
import cv2
import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


LOW_DIR = "/root/data/BUCHONG/Car"  

METHODS = {
    "CIDNet":      ("/root/data/darkface_exdark/CIDNet/results/Car", "_cidnet"),
    "DarkIR":      ("/root/data/darkface_exdark/DarkIR/results/Car", "_darkir"),
    "DMFourLLIE":  ("/root/data/darkface_exdark/DMFourLLIE/results/Car", "_dmfourllie"),
    "sfhformer":   ("/root/data/darkface_exdark/sfhformer/results/Car", "_sfhformer"),
    "ou":          ("/root/data/darkface_exdark/ou/results/Car", "_ou"),
    "RetinexNet":  ("/root/data/darkface_exdark/RetinexNet/results/Car", "_RetinexNet"),
    "EnlightenGAN":  ("/root/data/darkface_exdark/RetinexNet/results/Car", "_EnlightenGAN"),
    "FourLLIE":  ("/root/data/darkface_exdark/RetinexNet/results/Car", "_FourLLIE"),
    "RUAS":  ("/root/data/darkface_exdark/RUAS/results/Car", "_RUAS"),
    "LLFormer":  ("/root/data/darkface_exdark/LLFormer/results/Car", "_LLFormer"),
}

SAVE_ROOT = "./detection_car"
CONF_THRESH = 0.3
# ========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}")
model = YOLO("./yolov8n.pt").to(device)


save_low = os.path.join(SAVE_ROOT, "Car_low")
os.makedirs(save_low, exist_ok=True)
print("begin")

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

    enh_map = {f[:10]: f for f in os.listdir(enh_dir) if f.endswith(('jpg', 'png'))}

    for name in img_names:
        prefix = name[:10]
        if prefix not in enh_map: continue

        enh_img = cv2.imread(os.path.join(enh_dir, enh_map[prefix]))
        if enh_img is None: continue

        det_img = model(enh_img, conf=CONF_THRESH, verbose=False)[0].plot()

   
        save_name = f"{prefix}{suffix}.jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), det_img)

print("all")