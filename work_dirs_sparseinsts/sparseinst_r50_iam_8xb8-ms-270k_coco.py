auto_scale_lr = dict(base_batch_size=64, enable=True)
backend = 'pillow'
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.SparseInst.sparseinst',
    ])
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True, test_out_dir='vis', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs_sparseinsts/best_coco_segm_mAP_iter_8500.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=0,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    criterion=dict(
        assigner=dict(alpha=0.8, beta=0.2, type='SparseInstMatcher'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction='sum',
            type='FocalLoss',
            use_sigmoid=True),
        loss_dice=dict(
            eps=5e-05,
            loss_weight=2.0,
            reduction='sum',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=2,
        type='SparseInstCriterion'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        in_channels=258,
        ins_conv=4,
        ins_dim=256,
        kernel_dim=128,
        mask_conv=4,
        mask_dim=256,
        num_classes=2,
        num_masks=100,
        output_iam=False,
        scale_factor=2.0,
        type='BaseIAMDecoder'),
    encoder=dict(
        in_channels=[
            512,
            1024,
            2048,
        ],
        out_channels=256,
        type='InstanceContextEncoder'),
    test_cfg=dict(mask_thr_binary=0.45, score_thr=0.005),
    type='SparseInst')
optim_wrapper = dict(
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=10000,
        gamma=0.1,
        milestones=[
            8000,
            9000,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        metainfo=dict(classes=(
            'lk',
            'lk-c',
        )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    640,
                    853,
                ),
                type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR@300',
        'AR@1000',
    ],
    outfile_prefix='./work_dirs_sparseinst/test_results',
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(backend='pillow', keep_ratio=True, scale=(
        640,
        853,
    ), type='Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=(
            'lk',
            'lk-c',
        )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                backend='pillow',
                keep_ratio=True,
                scales=[
                    (
                        416,
                        853,
                    ),
                    (
                        448,
                        853,
                    ),
                    (
                        480,
                        853,
                    ),
                    (
                        512,
                        853,
                    ),
                    (
                        544,
                        853,
                    ),
                    (
                        576,
                        853,
                    ),
                    (
                        608,
                        853,
                    ),
                    (
                        640,
                        853,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        backend='pillow',
        keep_ratio=True,
        scales=[
            (
                416,
                853,
            ),
            (
                448,
                853,
            ),
            (
                480,
                853,
            ),
            (
                512,
                853,
            ),
            (
                544,
                853,
            ),
            (
                576,
                853,
            ),
            (
                608,
                853,
            ),
            (
                640,
                853,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        metainfo=dict(classes=(
            'lk',
            'lk-c',
        )),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    640,
                    853,
                ),
                type='Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR@300',
        'AR@1000',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs_sparseinsts'
