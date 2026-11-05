import numpy as np
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as compare_ssim
import torch
import torch.nn.functional as F


def to_ssim_skimage(dehaze, gt):
  dehaze_list = torch.split(dehaze, 1, dim=0)
  gt_list = torch.split(gt, 1, dim=0)

  dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in
                    range(len(dehaze_list))]
  gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
  ssim_list = [compare_ssim(dehaze_list_np[ind], gt_list_np[ind], data_range=1, multichannel=True) for ind in
               range(len(dehaze_list))]

  return ssim_list

def _convert_input_type_range(img):

  img_type = img.dtype
  img = img.astype(np.float32)
  if img_type == np.float32:
    pass
  elif img_type == np.uint8:
    img /= 255.
  else:
    raise TypeError('The img type should be np.float32 or np.uint8, '
                    f'but got {img_type}')
  return img


def _convert_output_type_range(img, dst_type):

  if dst_type not in (np.uint8, np.float32):
    raise TypeError('The dst_type should be np.float32 or np.uint8, '
                    f'but got {dst_type}')
  if dst_type == np.uint8:
    img = img.round()
  else:
    img /= 255.

  return img.astype(dst_type)

def rgb2ycbcr(img, y_only=False):

  img_type = img.dtype
  img = _convert_input_type_range(img)
  if y_only:
    out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
  else:
    out_img = np.matmul(img,
                        [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]) + [16, 128, 128]
  out_img = _convert_output_type_range(out_img, img_type)
  return out_img


def to_y_channel(img):
  """Change to Y channel of YCbCr.
  Args:
    img (ndarray): Images with range [0, 255].
  Returns:
    (ndarray): Images with range [0, 255] (float type) without round.
  """
  img = img.astype(np.float32) / 255.
  img = rgb2ycbcr(img, y_only=True)
  img = img[..., None]
  return img * 255.


def calculate_psnr_torch(img1, img2):
  b, c, h, w = img1.shape
  v = torch.tensor([[65.481/255], [128.553/255], [24.966/255]]).cuda()
  img1 = torch.mm(img1.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img2 = torch.mm(img2.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img1 = img1.reshape(b, h, w, -1)
  img2 = img2.reshape(b, h, w, -1)
  mse_loss = F.mse_loss(img1, img2, reduction='none').mean((1, 2, 3))
  psnr_full = 10 * torch.log10(1 / mse_loss).mean()
  sim = ssim(img1.permute(0, 3, 1, 2), img2.permute(0, 3, 1, 2), data_range=1, size_average=False).mean()

  return psnr_full, sim

import math

def pixel_weight(height, width, sample, beta):
    h = height
    w = width
    k = sample
    alpha = (1 - beta / k) / (beta - 1)

    w_weight = np.empty([h, w], dtype=float)
    h_weight = np.empty([h, w], dtype=float)

    if w > 2 * k:
        for x in range(w):
            if x < k:
                weight = ((1 / (x+1)) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            elif x < (w-k):
                weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w-x) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k)/k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
    else:
        for x in range(w):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k) / k + w * alpha)
                for i in range(h): w_weight[i][x] = weight
            else:
                weight = (1 / (w - x) + alpha) / (2 * (math.log(k)+0.5772) + (w - 2 * k) / k + w * alpha)
                for i in range(h): w_weight[i][x] = weight


    if h > 2 * k:
        for x in range(h):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            elif x < (h - k):
                weight = (1 / k + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h - x) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
    else:
        for x in range(w):
            if x < k:
                weight = ((1 / (x + 1)) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight
            else:
                weight = (1 / (h - x) + alpha) / (2 * (math.log(k)+0.5772) + (h - 2 * k) / k + h * alpha)
                h_weight[x][:] = weight


    total_weight = (w_weight + h_weight) / (h+w)

    print(total_weight)

    return total_weight


import cv2

def calculate_psnr(img1,
                   img2,
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.
        return 20. * np.log10(max_value / np.sqrt(mse))

    return _psnr(img1, img2)


def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out


def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d


def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)

    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window, borderType=bt)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   test_y_channel=False,
                   ssim3d=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """


    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)


    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []





        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)







        return np.array(ssims).mean()

    return _cal_ssim(img1, img2)


