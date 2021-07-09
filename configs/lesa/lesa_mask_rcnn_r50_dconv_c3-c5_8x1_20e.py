_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_1024.py',
    '../_base_/schedules/schedule_20e.py', 
    '../_base_/default_runtime.py',
]

optimizer = dict(lr=0.01)

model = dict(
    
    pretrained=\
    './checkpoints/lesa_pretrained_imagenet/'+\
    'lesa_resnet50_pretrained/'+\
    'lesa_resnet50/'+\
    'checkpoint.pth',

    backbone=dict(
        type='ResNet',
        strides=(1,2,2,2),
        wrn=False,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),

        stage_spatial_res=[256, 128, 64, 32], # 1024: [256, 128, 64, 32], 1280: [320, 160, 80, 40]
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
    ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
    ),
)

data_root = 'data/coco/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    # test=dict(
    #     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
    #     img_prefix=data_root + 'test2017/'),    
)

