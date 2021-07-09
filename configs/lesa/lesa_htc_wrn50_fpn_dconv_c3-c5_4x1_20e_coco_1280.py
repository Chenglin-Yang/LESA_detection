_base_ = '../htc/htc_r50_fpn_1x_coco_1280.py'

# optimizer
optimizer = dict(lr=0.005)

model = dict(
    
    pretrained=\
    './checkpoints/lesa_pretrained_imagenet/'+\
    'lesa_wrn50_pretrained/'+\
    'lesa_wrn50/'+\
    'checkpoint.pth',
    
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        strides=(1,2,2,2),
        wrn=True,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True), 

        stage_spatial_res=[320, 160, 80, 40], # 1024: [256, 128, 64, 32], 1280: [320, 160, 80, 40]
        stage_with_first_conv = [True, True, True, False],
        lesa=dict(
            type='LESA',
            with_cp_UB_terms_only=True, # cp used on the unary and binary terms only.
            pe_type='detection_qr', # ('classification', 'detection_qr')
            groups = 8,
            df_channel_shrink = [2], # df: dynamic fusion
            df_kernel_size = [1,1],
            df_group = [1,1],
        ),
        stage_with_lesa = (False, False, True, True),
    )
)

# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize', img_scale=(1280, 1280), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]

data_root = 'data/coco/'
data = dict(
    samples_per_gpu=1, workers_per_gpu=1, train=dict(pipeline=train_pipeline),
    # test=dict(
    #     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    #     img_prefix=data_root + 'test2017/'),    
)

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)