ann_file_train = 'data/keyframes_pilot_cockpit/annotations/train_annotations.csv'
ann_file_val = 'data/keyframes_pilot_cockpit/annotations/val_annotations.csv'
auto_scale_lr = dict(enable=False)
data_root = 'data/keyframes_pilot_cockpit/frames'
dataset_type = 'KeyFrameClipDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10, max_keep_ckpts=3, save_best='auto',
        type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        channel_ratio=8,
        fast_pathway=dict(
            base_channels=8,
            conv1_kernel=(
                5,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            enable_stta=True,
            lateral=False,
            pool1_stride_t=1,
            pretrained=None,
            spatial_strides=(
                1,
                2,
                2,
                1,
            ),
            stta_kernel_size=7,
            stta_stages=[
                True,
                True,
                True,
                True,
            ],
            type='resnet3d_stta'),
        fast_stta_enabled=True,
        fast_stta_kernel_size=7,
        fast_stta_stages=[
            True,
            True,
            True,
            True,
        ],
        pretrained=None,
        resample_rate=4,
        slow_pathway=dict(
            conv1_kernel=(
                1,
                7,
                7,
            ),
            conv1_stride_t=1,
            depth=50,
            dilations=(
                1,
                1,
                1,
                1,
            ),
            inflate=(
                0,
                0,
                1,
                1,
            ),
            lateral=True,
            pool1_stride_t=1,
            pretrained=None,
            spatial_strides=(
                1,
                2,
                2,
                1,
            ),
            type='resnet3d'),
        speed_ratio=4,
        type='ResNet3dSlowFastWithSTTA'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=2304,
        init_std=0.01,
        num_classes=8,
        spatial_type='avg',
        type='SlowFastHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    type='Recognizer3D')
num_classes = 8
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=160,
        begin=0,
        by_epoch=True,
        end=150,
        eta_min=0,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/keyframes_pilot_cockpit/annotations/val_annotations.csv',
        clip_len=32,
        data_prefix=dict(img='data/keyframes_pilot_cockpit/frames'),
        filename_tmpl='img_{:05}.jpg',
        frame_interval=1,
        num_clips=1,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=1,
                temporal_jitter=False,
                test_mode=True,
                type='SampleKeyFrameClips'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='KeyFrameClipDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(
        clip_len=32,
        frame_interval=1,
        temporal_jitter=False,
        test_mode=True,
        type='SampleKeyFrameClips'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=150, type='EpochBasedTrainLoop', val_begin=1, val_interval=2)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file=
        'data/keyframes_pilot_cockpit/annotations/train_annotations.csv',
        clip_len=32,
        data_prefix=dict(img='data/keyframes_pilot_cockpit/frames'),
        filename_tmpl='img_{:05}.jpg',
        frame_interval=1,
        num_clips=1,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=1,
                jitter_range=2,
                temporal_jitter=True,
                test_mode=False,
                type='SampleKeyFrameClips'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=False,
        type='KeyFrameClipDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        clip_len=32,
        frame_interval=1,
        jitter_range=2,
        temporal_jitter=True,
        test_mode=False,
        type='SampleKeyFrameClips'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/keyframes_pilot_cockpit/annotations/val_annotations.csv',
        clip_len=32,
        data_prefix=dict(img='data/keyframes_pilot_cockpit/frames'),
        filename_tmpl='img_{:05}.jpg',
        frame_interval=1,
        num_clips=1,
        pipeline=[
            dict(
                clip_len=32,
                frame_interval=1,
                temporal_jitter=False,
                test_mode=True,
                type='SampleKeyFrameClips'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='KeyFrameClipDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(
        clip_len=32,
        frame_interval=1,
        temporal_jitter=False,
        test_mode=True,
        type='SampleKeyFrameClips'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/SlowfastSTTA_r50_2xb8-448-32x1x1-150e_pilot_cockpit-rgb-V2'
