_base_ = [
    'default_runtime.py', './thumos14_feat.py'
]

enc_layers = 4
dec_layers = 4
dim_feature = 2048
dim_feedforward = 1024
dropout = 0.1

cls_loss_coef = 2
seg_loss_coef = 5
iou_loss_coef = 2

max_per_img = 200
lr = 0.0001

# model setting
model = dict(
    type='TDTR',
    num_queries=200,
    with_box_refine=True,
    as_two_stage=False,
    query_from_enc=False,
    query_pos_from_enc=False,
    num_feature_levels=4,
    data_preprocessor=dict(type='DetDataPreprocessor'),
    backbone=dict(type='PseudoBackbone', multi_scale=False),  # No backbone since we use pre-extracted features.
    neck=[
        dict(type='Flatten', start_dim=2),  # (N, C, 1, T) to (N, C, T)
        dict(
            type='TemporalDownSampler',
            num_levels=4,
            in_channels=dim_feature,
            out_channels=dim_feature,
            conv_type='Conv1d',
            kernel_sizes=3,
            strides=2,
            paddings=1,
            out_indices=(0, 1, 2, 3)),
        dict(type='Unflatten', dim=-1, unflattened_size=(1, -1)),  # (N, C, T) to (N, C, 1, T) mimic NCHW
        dict(
            type='ChannelMapper',
            in_channels=[dim_feature] * 4,
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)],
    encoder=dict(
        num_layers=enc_layers,
        layer_cfg=dict(
            self_attn_cfg=dict(
                num_levels=4,  # Using multi-level features
                embed_dims=256,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout)),
        memory_fuse=True),  # Using memory fusion
    decoder=dict(
        num_layers=dec_layers,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=dropout,
                batch_first=True),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,  # Using multi-level features
                dropout=dropout,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=256, normalize=True, offset=-0.5, temperature=10000),
    bbox_head=dict(
        type='TDTRHead',
        num_classes=20,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=cls_loss_coef),
        loss_bbox=dict(type='L1Loss', loss_weight=seg_loss_coef),
        loss_iou=dict(type='GIoU1dLoss', loss_weight=iou_loss_coef)),
    dn_cfg=dict(label_noise_scale=0.5, box_noise_scale=1.0,
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    # group_cfg=dict(dynamic=False, num_groups=5)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),  # from 6.0 to 2.0
                dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoU1dCost', iou_mode='giou', weight=2.0)])),  # from iou to giou
    test_cfg=dict(max_per_img=200)  # from 100 to 200 since window size is scaled
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    'sampling_offsets': dict(lr_mult=0.1),
                                    'reference_points': dict(lr_mult=0.1)}))
# learning policy
max_epochs = 12  # 16 for TadTR
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[10], gamma=0.1)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

train_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PadFeature', pad_length=256),
    dict(type='PackTADInputs', meta_keys=())
]
test_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PadFeature', pad_length=256),
    dict(type='PackTADInputs',
         meta_keys=('video_name', 'window_offset', 'fps', 'feat_stride', 'valid_len', 'overlap'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_val.json',
        feat_stride=4,
        pre_load_feat=False,
        window_size=256,
        window_stride=32,  # overlap=0.75
        iof_thr=0.75,
        skip_short=0.3,  # skip action annotations with duration less than 0.3 seconds
        skip_wrong=True,  # skip action annotations out of the range of video duration
        # data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432'),
        data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features'),
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
        feat_stride=4,
        window_size=256,
        window_stride=64,  # overlap=0.25
        skip_short=False,
        skip_wrong=True,
        # data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432'),
        data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='TadMetric', merge_windows=True, iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
                     nms_cfg=dict(type='nms', iou_thr=0.6))  # 0.4 for TadTR
test_evaluator = val_evaluator
