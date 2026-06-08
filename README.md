# SFGCNet

Official PyTorch implementation of **SFGCNet: A Low-Light Image Enhancement Network with Spatial-Frequency Collaborative Learning and Global Context Awareness**.

## Introduction

**SFGCNet** is designed for low-light image enhancement under challenging illumination and noise conditions. 

## Environment

The experiments are conducted under the following environment:

```text
Ubuntu 22.04
Python 3.10.19
PyTorch
CUDA
```

Install the basic dependencies:

```bash
pip install numpy opencv-python pillow tqdm
```

Other dependencies can be installed according to the released code requirements.

## Datasets

We evaluate SFGCNet on paired and unpaired low-light image enhancement datasets.

Paired datasets:

- LOL-v1
- LOLv2-Real
- LOLv2-Synthetic

Unpaired datasets:

- LIME
- DICM
- NPE
- MEF
- VV

Additional evaluation datasets:

- ExDark for low-light object detection evaluation
- UIEB for underwater image enhancement evaluation

Please download the datasets from their official sources and organize them according to your training and testing scripts.

A typical paired dataset structure is:

```text
dataset/
├── train/
│   ├── low/
│   └── high/
└── test/
    ├── low/
    └── high/
```

## Training Settings

SFGCNet is optimized using the Adam optimizer with:

```text
beta1 = 0.9
beta2 = 0.999
initial learning rate = 1e-4
minimum learning rate = 1e-6
scheduler = cosine annealing
```

For LOL-v1 and LOLv2-Synthetic, we use random cropped patches of size `128 × 128`. The batch size is set to `16` or `24`, and the model is trained for `1000` epochs.

For LOLv2-Real, we use patches of size `256 × 256` with a batch size of `10`. A fixed-iteration strategy is adopted, where each epoch contains `20` iterations and the model is trained for `5000` epochs, resulting in `100000` optimization steps.

Validation is performed regularly during training, and the best checkpoint is selected according to validation performance.

## Evaluation

For paired datasets, we report:

- PSNR
- SSIM
- LPIPS

For unpaired datasets, we report:

- BRISQUE

We also conduct cross-domain evaluations on ExDark and UIEB to validate the generalization ability of SFGCNet in low-light object detection and underwater image enhancement scenarios.

## Acknowledgement

We thank the authors of the public low-light image enhancement datasets and related open-source projects.