_base_ = ['./default_runtime.py']

# model settings
model = dict(type='SingleStageDetector',
             backbone=dict(type='SlowOnly', freeze_bn=True, freeze_bn_affine=True),
             neck=[
                 dict(type='MaxPool3d', kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                 dict(type='TemporalDownSampler',
                      num_levels=5,
                      in_channels=2048,
                      out_channels=512,
                      conv_type='Conv3d',
                      kernel_sizes=(3, 1, 1),
                      strides=(2, 1, 1),
                      paddings=(1, 0, 0),
                      out_indices=(0, 1, 2, 3, 4)),  # the index-0 denotes the input feature
                 dict(type='AdaptiveAvgPool3d', output_size=(None, 1, 1)),  # (N, C, T, H, W) to (N, C, T, 1, 1)
                 dict(type='Flatten', start_dim=2),  # (N, C, T, 1, 1) to (N, C, T)
                 dict(type='FPN',
                      in_channels=[2048, 512, 512, 512, 512],
                      out_channels=256,
                      num_outs=5,
                      conv_cfg=dict(type='Conv1d'),
                      norm_cfg=dict(type='SyncBN'))
             ],
             bbox_head=dict(type='RetinaHead1D',
                            num_classes=20,
                            in_channels=256,
                            conv_cfg=dict(type='Conv1d'),
                            norm_cfg=dict(type='SyncBN'),
                            anchor_generator=dict(
                                type='AnchorGenerator',
                                octave_base_scale=2,
                                scales_per_octave=5,
                                ratios=[1.0],
                                strides=[2, 4, 8, 16, 32]),
                            bbox_coder=dict(
                                type='DeltaXYWHBBoxCoder',
                                target_means=[.0, .0, .0, .0],
                                target_stds=[1.0, 1.0, 1.0, 1.0]),
                            reg_decoded_bbox=True,
                            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                            loss_bbox=dict(type='DIoU1dLoss', loss_weight=1.0),  # DIoU is computed on x-axis only
                            init_cfg=dict(type='Normal', layer='Conv1d', std=0.01,
                                          override=dict(type='Normal', name='retina_cls', std=0.01, bias_prob=0.01))),
             data_preprocessor=dict(type='mmaction.ActionDataPreprocessor',
                                    mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375],
                                    format_shape='NCTHW'),
             train_cfg=dict(assigner=dict(type='MaxIoUAssigner',
                                          pos_iou_thr=0.6,
                                          neg_iou_thr=0.4,
                                          min_pos_iou=0,
                                          ignore_iof_thr=-1,
                                          ignore_wrt_candidates=True,
                                          iou_calculator=dict(type='BboxOverlaps1D')),
                            allowed_border=-1,
                            pos_weight=-1,
                            debug=False),
             test_cfg=dict(nms_pre=300, score_thr=0.005))

# dataset settings
data_root = 'my_data/thumos14'  # Root path to data for training
data_prefix_train = 'rawframes/val'  # path to data for training
data_prefix_val = 'rawframes/test'  # path to data for validation and testing

clip_len = 192
frame_interval = 5
img_shape = (160, 160)
img_shape_test = (160, 160)

train_pipeline = [
    dict(type='RandomSlice',
         window_size=clip_len * frame_interval,
         frame_interval=frame_interval,
         iof_thr=0.75),
    dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=(180, -1), keep_ratio=True),
    dict(type='mmaction.RandomCrop', size=img_shape[0]),  # mmaction2 RandomCrop only supports square crop
    dict(type='mmaction.Flip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion3d',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Rotate3d',
         limit=(-45, 45),
         border_mode='reflect_101',
         prob=0.5),
    dict(type='Pad3d', size=(clip_len, *img_shape)),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs', meta_keys=())
]

val_pipeline = [
    dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=(180, -1), keep_ratio=True),
    dict(type='mmaction.CenterCrop', crop_size=img_shape_test),
    dict(type='Pad3d', size=(clip_len, *img_shape_test)),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs',
         meta_keys=('video_name', 'fps', 'window_offset', 'valid_len', 'frame_interval', 'overlap'))
]

train_dataloader = dict(  # Config of train dataloader
    batch_size=16,  # Batch size of each single GPU during training
    num_workers=6,  # Workers to pre-fetch data for each single GPU during training
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(  # Config of train dataset
        type='THUMOS14Dataset',
        filename_tmpl='img_{:05}.jpg',
        ann_file='annotations/mmtad/thumos14_val.json',
        data_root=data_root,  # Root path to data, including both frames and ann_file
        data_prefix=dict(frames=data_prefix_train),  # Prefix of specific data, e.g., frames and ann_file
        pipeline=train_pipeline))
val_dataloader = dict(  # Config of validation dataloader
    batch_size=1,  # Batch size of each single GPU during validation
    num_workers=6,  # Workers to pre-fetch data for each single GPU during validation
    persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing
    dataset=dict(  # Config of validation dataset
        type='THUMOS14Dataset',
        window_size=int(clip_len * frame_interval),
        window_stride=int(clip_len * frame_interval) * 0.25,
        frame_interval=frame_interval,
        filename_tmpl='img_{:05}.jpg',
        ann_file='annotations/mmtad/thumos14_test.json',
        data_root=data_root,
        data_prefix=dict(frames=data_prefix_val),  # Prefix of specific data components
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = val_dataloader

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1200, val_begin=1, val_interval=100)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=40, convert_to_iter_based=True),
    dict(type='CosineRestartLR', periods=[100] * 12, restart_weights=[1] * 12, eta_min=1e-4, by_epoch=True,
         begin=40, end=1200, convert_to_iter_based=True)
]
# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# evaluation settings
val_evaluator = dict(type='TadMetric',
                     merge_windows=True,
                     iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
                     max_per_video=1200,
                     nms_cfg=dict(type='nms', iou_thr=0.5),
                     voting_cfg=dict(iou_thr=0.5))
test_evaluator = val_evaluator

default_hooks = dict(logger=dict(interval=20, interval_exp_name=1000), checkpoint=dict(interval=100, max_keep_ckpts=12))
# save memory
efficient_conv_bn_eval = ['backbone']  # only work for slowonly
