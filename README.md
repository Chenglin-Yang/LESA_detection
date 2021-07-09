# LESA_detection

<div align="center">
  <img src="Images/figure1_v4.png" width="800px" />
</div>

## Introduction

This repository contains the official implementation of [Locally Enhanced Self-Attention: Rethinking Self-Attention as Local and Context Terms](https://arxiv.org/abs/).
The code for object detection is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please refer to [LESA](https://github.com/Chenglin-Yang/LESA) for the whole project.

## Installation

Please refer to [mmdetection readme](README_mmdet.md) for installation and usage. This code is tested with pytorch 1.7.1. After preparing the environment, please issue: 
```bash
pip install einops
```

## Main Results on COCO test-dev

| Method    | Backbone          | Config | Pretrained | Model | Box AP | Mask AP |
|:-----------:|:-----------------:|--------------|--------------|--------------|:------------:|:------------:|
| Mask-RCNN | LESA_ResNet50         | [File Link](configs/lesa/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e.py) | [Download](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/cyang76_jh_edu/EsV_fGZY-uhEkciwVckp4c8BlInA1GFv7gett1_LOZ0vFg?e=g1If75) | [Download](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/cyang76_jh_edu/Egpo87VmMmlEg0jY_KHYAJsBS7EFDJ4YxJ2zhkTxcJCzWg?e=SGSKmx) | 44.2 | 39.6 |
| HTC | LESA_WRN50 | [File Link](configs/lesa/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280.py) | [Download](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/cyang76_jh_edu/ElagAKEgXttArEbtVR6NpmEBWAZN0pNE5Q6MMXEJZ27VHg?e=dIdwAI) | [Download](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/cyang76_jh_edu/EmSPz8ToSK5GuYyWELj3Y0QBwP3Q_Jd4FhK1WDvf2FuADw?e=xRsbl5) | 50.5 | 44.4 |

## Citing LESA

If you find LESA is helpful in your project, please consider citing our paper.

```BibTeX
@article{yang2021locally,
  title={Locally Enhanced Self-Attention: Rethinking Self-Attention as Local and Context Terms},
  author={Yang, Chenglin and Qiao, Siyuan and Kortylewski, Adam and Yuille, Alan},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```

## Usage

### Training:
+ Before training, please download the pretrained weights (folders) and put them in ./checkpoints/lesa_pretrained_imagenet/.
```bash
# LESA_ResNet50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=-29700 \
bash ./tools/dist_train.sh \
./configs/lesa/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e.py \
8 
```
```bash
# LESA_WRN50
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=-29800 \
bash ./tools/dist_train.sh \
./configs/lesa/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280.py \
4
```

### Validation on COCO val:
+ Before validation, please download the checkpoints (folders) and put them in ./work_dirs/.
```bash
# LESA_ResNet50
# Box AP: 44.0 -- Mask AP: 39.2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=-30001 \
./tools/dist_test.sh \
./configs/lesa/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e.py \
work_dirs/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e/epoch_20.pth \
8 \
--out work_dirs/test.pkl \
--eval bbox segm
```
```bash
# LESA_WRN50
# Box AP: 50.1 -- Mask AP: 43.9
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=-30001 \
./tools/dist_test.sh \
./configs/lesa/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280.py \
work_dirs/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280/epoch_20.pth \
8 \
--out work_dirs/test.pkl \
--eval bbox segm
```

### Generating the json file for testing on COCO test-dev:
+ Before testing, uncomment "test=dict(...)" in the config files.
```bash
# LESA_ResNet50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=-30001 \
./tools/dist_test.sh \
./configs/lesa/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e.py \
work_dirs/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e/epoch_20.pth \
8 \
--format-only \
--options "jsonfile_prefix=./work_dirs/lesa_mask_rcnn_r50_dconv_c3-c5_8x1_20e"
```
```bash
# LESA_WRN50
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=-30001 \
./tools/dist_test.sh \
./configs/lesa/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280.py \
work_dirs/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280/epoch_20.pth \
8 \
--format-only \
--options "jsonfile_prefix=./work_dirs/lesa_htc_wrn50_fpn_dconv_c3-c5_4x1_20e_coco_1280"
```

## Credits

This project is based on [axial-deeplab](https://github.com/csrhddlam/axial-deeplab) and [mmdetection](https://github.com/open-mmlab/mmdetection).

Relative position embedding is based on [bottleneck-transformer-pytorch](https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py)

ResNet is based on [pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). Classification helper functions are based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification).

