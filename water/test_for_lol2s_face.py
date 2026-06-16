import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from models import *
from collections import OrderedDict

try:
    import pyiqa
except ImportError:
    raise ImportError("Please install pyiqa first: pip install pyiqa")


class SimpleDataset(Dataset):
    def __init__(self, folder_path):
        self.files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return {'source': img, 'name': os.path.basename(path)}


def test_on_folder(data_dir, save_dir, network, niqe_metric, bris_metric, device):
    network.eval()
    os.makedirs(save_dir, exist_ok=True)

    dataset = SimpleDataset(data_dir)
    if len(dataset) == 0:
        print(f"No images found in {data_dir}")
        return np.nan, np.nan

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    n_scores = []
    b_scores = []

    for batch in loader:
        source = batch['source'].to(device)
        name = batch['name'][0]

        h, w = source.shape[2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        pad_mode = 'reflect' if ph < h and pw < w else 'replicate'
        source = F.pad(source, (0, pw, 0, ph), mode=pad_mode)

        with torch.inference_mode():
            outputs = network(source)

            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs

            output = torch.clamp(output, 0, 1)
            output = output[:, :, :h, :w]

            try:
                cur_niqe = niqe_metric(output).item()
                raw_brisque = bris_metric(output).item()
                cur_brisque = raw_brisque
            except Exception as e:
                print(f"{name}, metric error: {e}")
                cur_niqe, cur_brisque, raw_brisque = np.nan, np.nan, np.nan

        n_scores.append(cur_niqe)
        b_scores.append(cur_brisque)

        file_base, file_ext = os.path.splitext(name)

        niqe_str = f"{cur_niqe:.2f}" if not np.isnan(cur_niqe) else "nan"
        bris_str = f"{cur_brisque:.2f}" if not np.isnan(cur_brisque) else "nan"

        save_name = f"{file_base}_lowlight_NIQE_{niqe_str}_BRIS_{bris_str}{file_ext}"

        res_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, save_name), cv2.cvtColor(res_np, cv2.COLOR_RGB2BGR))

        log_msg = f"  -> Saved: {save_name}"
        if not np.isnan(raw_brisque) and raw_brisque < 0:
            log_msg += f" ({raw_brisque:.2f})"
        print(log_msg)

        del source, outputs, output
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return np.nanmean(n_scores), np.nanmean(b_scores)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'sfhformer_lol_s'

    model_zoo = {
        'sfhformer_lol_s': sfhformer_lol_s,
    }

    if model_name not in model_zoo:
        raise ValueError(f"Unknown model name: {model_name}")

    network = model_zoo[model_name]()

    ckpt_path = '/home/leng/data/darkface_exdark/ou/saved_models/lowlight/sfhformer_lol_strain_lolv2s_best.pth'

    if not os.path.exists(ckpt_path):
        print(f"{ckpt_path}")
        exit()

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    network.load_state_dict(new_state_dict)
    network = network.to(device)

    print(f"{ckpt_path}")

    niqe_tool = pyiqa.create_metric('niqe', device=device)
    bris_tool = pyiqa.create_metric('brisque', device=device)

    base_path = '/home/leng/data/qtacefz/data/BUCHONG'
    sub_datasets = ['track1.2_test_sample', 'Car', 'Dog']

    results_map = {}
    print("\n开始测试")

    for ds in sub_datasets:
        folder = os.path.join(base_path, ds)

        if not os.path.exists(folder):
            print(f"{ds}")
            continue

        print(f"\n>>> Dataset: {ds}")
        save_path = f'./results/{ds}'
        m_niqe, m_bris = test_on_folder(folder, save_path, network, niqe_tool, bris_tool, device)
        results_map[ds] = (m_niqe, m_bris)

    if results_map:
        avg_niqe = np.nanmean([v[0] for v in results_map.values()])
        avg_brisque = np.nanmean([v[1] for v in results_map.values()])

        print("\n" + "=" * 50)
        print(f"{'Dataset':<20} | {'NIQE (↓)':<10} | {'BRISQUE (↓)':<10}")
        print("-" * 50)

        for ds, (ni, br) in results_map.items():
            ni_str = f"{ni:<10.4f}" if not np.isnan(ni) else f"{'nan':<10}"
            br_str = f"{br:<10.4f}" if not np.isnan(br) else f"{'nan':<10}"
            print(f"{ds:<20} | {ni_str} | {br_str}")

        print("-" * 50)
        print(f"{'Total Avg':<20} | {avg_niqe:<10.4f} | {avg_brisque:<10.4f}")
        print("=" * 50)