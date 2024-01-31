_base_ = [
    'default_runtime.py', './thumos14_feat.py'
]
custom_imports = dict(imports=['my_modules'], allow_failed_imports=False)

# TadTR (based on DeFormableDETR) setting: (DINO, TadTR)
enc_layers = 4  # 6, 4
dec_layers = 4  # 6, 4
dim_feedforward = 1024  # 2048, 1024
dropout = 0.1  # 0.0, 0.1

act_loss_coef = 4  # NA, 4
cls_loss_coef = 2  # 1.0, 2.0
seg_loss_coef = 5  # 5.0, 5.0
iou_loss_coef = 2  # 2.0, 2.0

max_per_img = 100  # 300, 100
lr = 0.0002  # 1e-4, 2e-4

# model setting
model = dict(
    type='TadTR',
    num_queries=40,  # num_matching_queries, should be smaller than the window size
    with_box_refine=True,
    as_two_stage=False,  # True for DeformableDETR
    num_feature_levels=1,
    data_preprocessor=dict(type='DetDataPreprocessor'),
    backbone=dict(type='PseudoBackbone'),  # No backbone since we use pre-extracted features.
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1),
    encoder=dict(
        num_layers=enc_layers,  # 6 for DeformableDETR
        layer_cfg=dict(
            self_attn_cfg=dict(
                num_levels=1,
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout))),
    decoder=dict(
        num_layers=dec_layers,  # 6 for DeformableDETR
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=dropout,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=1,
                dropout=dropout,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout)),
        post_norm_cfg=None),
    # offset=-0.5 for DeformableDETR;
    # the real num_feats is 128*2=256, 128 is just for the compatibility.
    positional_encoding=dict(num_feats=128, normalize=True, offset=0, temperature=10000),
    bbox_head=dict(
        type='TadTRHead',
        num_classes=20,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_coef),
        loss_bbox=dict(type='L1Loss', loss_weight=seg_loss_coef),
        loss_iou=dict(type='IOU1dLoss', mode='linear', loss_weight=iou_loss_coef)),  # -log(GIoU) for DeformableDETR
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=6.0),  # 2.0 for DeformableDETR
                dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoU1dCost', iou_mode='iou', weight=2.0)  # GIoU for DeformableDETR
            ])),
    test_cfg=dict(max_per_img=max_per_img))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.0001),  # 0.0001 by default
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    'sampling_offsets': dict(lr_mult=0.1),
                                    'reference_points': dict(lr_mult=0.1)
                                    }))

# learning policy
max_epochs = 16
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[14],
        gamma=0.1),
]

# dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

train_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'window_offset'))
]
test_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'window_offset', 'overlap'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_val.json',
        feat_stride=8,
        window_size=128,
        window_stride=32,
        iof_thr=0.75,
        skip_short=0.3,   # skip action annotations with duration less than 0.3 seconds
        skip_wrong=True,  # skip action annotations out of the range of video duration
        data_prefix=dict(feat='features/thumos_feat_TadTR_64input_8stride_2048'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_test.json',
        feat_stride=8,
        window_size=128,
        window_stride=96,
        skip_short=False,
        skip_wrong=True,
        data_prefix=dict(feat='features/thumos_feat_TadTR_64input_8stride_2048'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(
    type='TH14Metric',
    merge_windows=True,
    metric='mAP',
    iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
    nms_cfg=dict(type='nms', iou_thr=0.4))
test_evaluator = val_evaluator


