# model settings
model = dict(
    type='MaskRCNN',
    pretrained='modelzoo://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # frozen_stages=1,
        style='pytorch'),
    # resnet cjh
    # backbone=dict(
    #     type='ResNet_CJH',
    #     depth=101,
    #     num_stages=4,
    #     frozen_stages=1,
    #     style='pytorch'),

    ## densenet cjh
    # backbone=dict(
    #     type='DenseNet_CJH',),
        # num_init_features=136,
        # idx_version=121,
        # growth_rate=20,
        # bn_size=4,
        # drop_rate=0),
    # backbone=dict(
    #     type='DenseNet',
    #     growthRate=12,
    #     depth=100,
    #     reduction=0.5,
    #     nClasses=2,
    #     bottleneck=True),
    # backbone=dict(
    #     type='DenseNet',    
    #     # num_input_features=3,
    #     # growth_rate=12,
    #     # bn_size=2,
    #     # drop_rate=0.5,
    #     # memory_efficient=False
    #     ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,  # coco:81
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=2,  # coco:81
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'ZMVISION_BOAT522_ENHANCED' # 'zmvision_boat522_enhanced'
# data_root = '/home/edison/tools/detectron/detectron/detectron/datasets/data/boat520_datasets/'
data_root = '/home/edison/tools/datasets/zmvision-datasets/bigShip-6-1/'
data_root_changguang = '/home/edison/tools/datasets/zmvision-datasets/newDatasets_Phd_He_0628/changguang/dataset0628/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root_changguang + 'annotations/changguang0628LAB_he_train_0701.json', # 'json-files/xian_train64.json'#  train-61.json
        img_prefix=data_root_changguang + 'imgfiles/train/',
        # ann_file=data_root +  'annotations/json-files/xian_test64.json',
        # img_prefix=data_root + 'img-files/',
        img_scale=(1024, 1024), # 800, 1333 833, 500
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root_changguang + 'annotations/changguang0628LAB_he_train.json',
        img_prefix=data_root_changguang + 'imgfiles/train/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root_changguang + 'annotations/changguang0628LAB_he_test_0701.json', # test-negative-61.json test-61.json, xian_test_520_521_temp.json test-negative-61.json
        img_prefix=data_root_changguang + 'imgfiles/test/',
        # ann_file=data_root +  'annotations/json-files/xian_test64.json',
        # img_prefix=data_root + 'img-files/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500, # 500
    warmup_ratio=1.0 / 3,
    # step=[75, 88])
    step=[160, 180])
    # step=[1, 2])
checkpoint_config = dict(interval=10)

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 200
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'OutPutDir/cg_07102028_' + 'epoch_' + str(total_epochs) + '_' + model["backbone"]["type"] 
load_from = None
resume_from = None
workflow = [('train', 1)]
