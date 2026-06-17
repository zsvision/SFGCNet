# SFGCNet

Official PyTorch implementation of **SFGCNet: Spatial-Frequency Collaborative Learning with Global Context Awareness for Robust Low-Light Image Enhancement**.

## Introduction

SFGCNet is designed for low-light image enhancement under challenging illumination and noise conditions. In addition to standard low-light restoration, it is also evaluated on underwater image enhancement and low-light object detection tasks, demonstrating its generalization ability across different degraded visual scenarios.

## Environment

The experiments are conducted under the following environment:

```text
Ubuntu 22.04
Python 3.10.19
CUDA 11.8
```

Create and configure the experimental environment by executing the following commands:

```bash
conda create -n lowlight python=3.10.19 -y
conda activate lowlight
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## Datasets

We evaluate WD-Mamba on paired and unpaired low-light image enhancement datasets.

**Paired datasets:**
* LOL-v1
* LOLv2-Real
* LOLv2-Synthetic

**Unpaired datasets:**
* LIME
* DICM
* NPE
* MEF
* VV

**Additional evaluation datasets:**

* ExDark for low-light object detection evaluation
* UIEB for underwater image enhancement evaluation

### Download Links
You can download all the prepared datasets via Baidu Netdisk:

Baidu Netdisk: [Download Link](https://pan.baidu.com/s/1LVYxVL8CIM2N0cPHYUZ0Kg) (Extraction code: `gv4d`)

### Directory Structure
Please download the datasets and organize them according to the training and testing scripts. A typical dataset structure is as follows:

```text
data/
├── LOLdataset/
│   ├── our485/
│   │   ├── Train/
│   │   └── Test/
│   └── eval15/
│       ├── Train/
│       └── Test/
├── LOLv2/
│   ├── Real_captured/
│   │   ├── Train/
│   │   └── Test/
│   └── Synthetic/
│       ├── Train/
│       └── Test/
├── Unpaired_Datasets/
│   ├── DICM/
│   ├── LIME/
│   ├── MEF/
│   ├── NPE/
│   └── VV/
├── Challenging-60/
├── UIEB/
└── track1.2_test_sample/
```

## Pretrained Models

We provide the pretrained weights of SFGCNet for quick evaluation. You can download them from Baidu Netdisk:

Baidu Netdisk:[Download Link](https://pan.baidu.com/s/1DrF0D5vxDCizRF6eIETP4g) (Extraction code: `nt88`)

## Training Settings

SFGCNet is optimized using the Adam optimizer with:

```text
beta1 = 0.9
beta2 = 0.999
initial learning rate = 1e-4
minimum learning rate = 1e-6
```

For LOL-v1 and LOLv2-Synthetic, we use random cropped patches of size `128 × 128`. The batch size is set to `16` or `24`, and the model is trained for `1000` epochs.

For LOLv2-Real, we use patches of size `256 × 256` with a batch size of `10`. A fixed-iteration strategy is adopted, where each epoch contains `20` iterations and the model is trained for `5000` epochs, resulting in `100000` optimization steps.

Validation is performed regularly during training, and the best checkpoint is selected according to validation performance.

## Evaluation Metrics

To comprehensively assess the performance and generalization capability of our proposed method, we conduct extensive evaluations on the primary task of Low-Light Image Enhancement, as well as two downstream applications: Underwater Image Enhancement and Object Detection. 

The detailed datasets and metrics used for each task are summarized below:

| Task                                          | Dataset(s)                                        | Evaluation Metrics |
| :-------------------------------------------- | :------------------------------------------------ | :----------------- |
| **Low-Light Image Enhancement**               | **Paired:** LOL-v1, LOL-v2-Real, LOL-v2-Synthetic | PSNR, SSIM, LPIPS  |
|                                               | **Unpaired:** LIME, DICM, NPE, MEF, VV            | BRISQUE            |
| **Underwater Image Enhancement (Downstream)** | Challenging-60                                    | UCIQE, NIQE        |
| **Object Detection (Downstream)**             | ExDark, DarkFace                                  | mAP@50,mAP@50:95   |

**Metrics Details:**

* **Primary Task: Low-Light Image Enhancement (Paired):** For paired datasets including **LOL-v1**, **LOL-v2-Real**, and **LOL-v2-Synthetic**, we employ **PSNR** and **SSIM** to measure pixel-level restoration accuracy and structural fidelity, alongside **LPIPS** to evaluate human perceptual similarity.
* **Primary Task: Low-Light Image Enhancement (Unpaired):** For unpaired real-world testing datasets including **LIME**, **DICM**, **NPE**, **MEF**, and **VV**, we utilize the no-reference quality metric **BRISQUE** to evaluate the naturalness and visual quality of enhanced images without ground truth.
* **Downstream Application: Underwater Image Enhancement:** To demonstrate the generalization of our method to other visually degraded environments, we evaluate it on underwater scenes. We use no-reference metrics **UCIQE** and **NIQE** on the **Challenging-60** dataset to assess color correction and contrast balance. For the **UIEB** dataset, we focus strictly on qualitative visual comparisons.
* **Downstream Application: Object Detection:** To validate that our enhancement effectively recovers semantic details for high-level machine vision, we evaluate object detection performance on the **ExDark** and **DarkFace** datasets, reporting the mean Average Precision (**mAP@50** and **mAP@50:95**).

## Acknowledgement

We thank the authors of the public low-light image enhancement datasets and related open-source projects.