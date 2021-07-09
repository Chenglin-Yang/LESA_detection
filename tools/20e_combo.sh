#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PORT=-29700 \
bash ./tools/dist_train.sh \
./configs/ub/mask_rcnn_r50_0510_344_20e.py \
4 \
| tee ./work_dirs/log/0510_344_20e

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash ./tools/dist_train.sh \
./configs/ub/mask_rcnn_r50_0508_338_20e.py \
4 \
| tee ./work_dirs/log/0508_338_20e

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash ./tools/dist_train.sh \
./configs/ub/mask_rcnn_r50_0508_339_20e.py \
4 \
| tee ./work_dirs/log/0508_339_20e



# bash ./tools/dist_train.sh \
# ./configs/ub/mask_rcnn_r50_0508_336.py \
# 4 \
# | tee ./work_dirs/log/0508_336



