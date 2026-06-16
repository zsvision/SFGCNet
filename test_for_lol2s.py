# -*- coding: utf-8 -*-
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import AverageMeter
from datasets.LoL_DataLoader import TestData_for_LOLv2Synthetic
from pytorch_msssim import ssim
from models import *
import random
import numpy as np
import cv2
import lpips 


def set_seed(seed=8001):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(8001)
# ============================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_lol_s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--exp', default='lowlight', type=str, help='experiment setting')
args = parser.parse_args()


save_img_root = '/root/our/LOLV2_Synthetic/lowlight/results'
os.makedirs(save_img_root, exist_ok=True)

def valid(val_loader_full, network):
    PSNR_full = AverageMeter()
    SSIM_full = AverageMeter()
    LPIPS_full = AverageMeter() 


    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).cuda()

    network.eval()

    img_idx = 0
    for batch in val_loader_full:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(0, 1)

            mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
            psnr_single = 10 * torch.log10(1 / mse_loss).item()  # 单张的PSNR
            psnr_full = 10 * torch.log10(1 / mse_loss).mean()
            PSNR_full.update(psnr_full.item(), source_img.size(0))

            ssim_single = ssim(output, target_img, data_range=1, size_average=False).item()  # 单张的SSIM
            ssim_full = ssim(output, target_img, data_range=1, size_average=False).mean()
            SSIM_full.update(ssim_full.item(), source_img.size(0))


            output_lpips = output * 2.0 - 1.0
            target_lpips = target_img * 2.0 - 1.0


            lpips_val = loss_fn_alex(output_lpips, target_lpips)
            lpips_single = lpips_val.item()  # 单张的LPIPS
            LPIPS_full.update(lpips_val.mean().item(), source_img.size(0))


            psnr_str = f"{psnr_single:.2f}"
            ssim_str = f"{ssim_single:.4f}"
            lpips_str = f"{lpips_single:.4f}"

            source_np = (source_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            target_np = (target_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            output_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)


            file_prefix = f"{img_idx:04d}lowlight_psnr{psnr_str}_ssim{ssim_str}_lpips{lpips_str}"
            
            cv2.imwrite(os.path.join(save_img_root, f"{file_prefix}_source.jpg"), source_bgr)
            cv2.imwrite(os.path.join(save_img_root, f"{file_prefix}_target.jpg"), target_bgr)
            cv2.imwrite(os.path.join(save_img_root, f"{file_prefix}_output.jpg"), output_bgr)

            comparison = np.hstack([source_bgr, output_bgr, target_bgr])
            cv2.imwrite(os.path.join(save_img_root, f"{file_prefix}_comparison.jpg"), comparison)

            img_idx += 1

    return PSNR_full.avg, SSIM_full.avg, LPIPS_full.avg 

if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    print(setting_filename)
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    device_index = [0]
    network = eval(args.model.replace('-', '_'))()
    network = nn.DataParallel(network, device_ids=device_index).cuda()
    

    ckpt_path = '/root/our/LOLV2_Synthetic/lowlight/saved_models/lowlight/lolv2s.pth'
    network.load_state_dict(torch.load(ckpt_path)['state_dict'])


    def worker_init_fn(worker_id):
        worker_seed = 8001 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    test_data_dir = '/root/our/data/LOLv2/Synthetic/Test'
    test_dataset = TestData_for_LOLv2Synthetic(8, test_data_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    print({save_img_root})
    avg_psnr, avg_ssim, avg_lpips = valid(test_loader, network)
    
    print(format(avg_psnr, avg_ssim, avg_lpips))
    print({save_img_root})