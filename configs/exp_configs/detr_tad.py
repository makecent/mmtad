_base_ = [
    '../default_runtime.py',
]
#%% 1. Optimizer settings
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
                     clip_grad=dict(max_norm=1.0, norm_type=2))
# learning policy
max_epochs = 30
param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(type='LinearLR',
         start_factor=1e-8,
         by_epoch=True,
         begin=0,
         end=5,
         convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        begin=5,
        end=max_epochs,
        eta_min=1e-8,
        by_epoch=True,
        convert_to_iter_based=True)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=5)

#%% 2. Dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

max_seq_len = 2304
train_pipeline = [
    dict(type='RandomSlice', window_size=max_seq_len, iof_thr=0.5),
    dict(type='LoadFeature'),
    dict(type='PadFeature', pad_length=max_seq_len),
    dict(type='PackTADInputs', meta_keys=())
]
test_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PadFeature', pad_length=max_seq_len, pad_length_divisor=32 * (19 // 2) * 2),
    dict(type='PackTADInputs',
         meta_keys=('video_name', 'fps', 'feat_stride', 'valid_len'))]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_val.json',
        feat_stride=4,
        pre_load_feat=True,
        skip_short=0.3,  # skip action annotations with duration less than 0.3 seconds
        skip_wrong=True,  # skip action annotations out of the range of video duration
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
        pre_load_feat=True,
        skip_short=False,
        skip_wrong=True,
        data_prefix=dict(feat='features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader


#%% 3. Model settings
model = dict(
    type='DETR_TAD',
    num_queries=200,
    data_preprocessor=dict(type='DetDataPreprocessor'),
    backbone=dict(type='PseudoBackbone', multi_scale=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=[2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=1),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)))),
    decoder=dict(  # DetrTransformerDecoder
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))),
        return_intermediate=True),
    positional_encoding=dict(num_feats=256, normalize=True),
    bbox_head=dict(
        type='DETR_TADHead',
        num_classes=20,
        embed_dims=256,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoU1dLoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.),
                dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoU1dCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=200))

val_evaluator = dict(type='TadMetric', iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7])
test_evaluator = val_evaluator
