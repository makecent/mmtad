_base_ = [
    'default_runtime.py',
]

enc_layers = 4
dec_layers = 4
# dim_feat = 2048
dim_feat = 768
dim_feedforward = 1024
dropout = 0.1

cls_loss_coef = 2
seg_loss_coef = 5
iou_loss_coef = 2

max_per_img = 200
lr = 0.0001

# model setting
model = dict(
    type='DITA',
    num_queries=200,
    with_box_refine=True,
    as_two_stage=False,
    query_from_enc=False,
    query_pos_from_enc=False,
    num_feature_levels=4,
    data_preprocessor=dict(
        type='mmaction.ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    backbone=dict(type='SlowOnly',
                  out_indices=(4,),
                  freeze=False,
                  freeze_bn=True,
                  freeze_bn_affine=True),
    # backbone=dict(type='VideoMAE_Base', freeze=False),
    neck=[
        dict(type='AdaptiveAvgPool3d', output_size=(None, 1, 1)),  # (N, C, T, H, W) to (N, C, T, 1, 1)
        dict(type='Flatten', start_dim=2),  # (N, C, T, 1, 1) to (N, C, T)
        dict(
            type='TemporalDownSampler',
            num_levels=4,
            in_channels=dim_feat,
            out_channels=512,
            conv_type='Conv1d',
            kernel_sizes=3,
            strides=2,
            paddings=1,
            out_indices=(0, 1, 2, 3)),
        dict(type='Unflatten', dim=-1, unflattened_size=(1, -1)),  # (N, C, T) to (N, C, 1, T) mimic NCHW
        dict(
            type='ChannelMapper',
            in_channels=[dim_feat, 512, 512, 512],
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
                num_levels=4,
                dropout=dropout,
                batch_first=True),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=dim_feedforward,
                ffn_drop=dropout)),
        post_norm_cfg=None),
    positional_encoding=dict(num_feats=256, normalize=True, offset=-0.5, temperature=10000),
    bbox_head=dict(
        type='DitaHead',
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
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoU1dCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=200)  # from 100 to 200 since window size is scaled
)

# optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    'sampling_offsets': dict(lr_mult=0.1),
                                    'reference_points': dict(lr_mult=0.1)}))
# learning policy
max_epochs = 6
param_scheduler = [dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[5], gamma=0.1)]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# dataset settings
dataset_type = 'THUMOS14Dataset'
data_root = 'my_data/thumos14/'

frame_interval = 4
window_size = 960
window_stride_train = 720  # overlap=0.75
window_stride_test = 720  # overlap=0.25
num_clips = 15  # 192/12=16 frame per clip
# window_size = 480
# window_stride_train = 360  # overlap=0.75
# window_stride_test = 360  # overlap=0.25
# num_clips = 6        # 96/6=16 frame per clip
# img_shape = (112, 112)
# img_shape_test = (128, 128)
img_shape = (224, 224)
img_shape_test = (224, 224)

train_pipeline = [
    dict(type='PseudoFrameDecode'),
    # dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=img_shape, keep_ratio=False),
    dict(type='mmaction.Flip', flip_ratio=0.5),
    dict(type='Pad3d', size=(window_size // frame_interval, *img_shape)),
    dict(type='TemporalSegment', num_clips=num_clips),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs', meta_keys=())
]
test_pipeline = [
    dict(type='PseudoFrameDecode'),
    # dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=img_shape_test, keep_ratio=False),
    dict(type='Pad3d', size=(window_size // frame_interval, *img_shape_test)),
    dict(type='TemporalSegment', num_clips=num_clips),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs',
         meta_keys=('video_name', 'window_offset', 'fps', 'frame_interval', 'valid_len', 'overlap'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/louis/thumos14_val.json',
        window_size=window_size,
        window_stride=window_stride_train,
        frame_interval=frame_interval,
        iof_thr=0.75,
        skip_short=0.3,  # skip action annotations with duration less than 0.3 seconds
        skip_wrong=True,  # skip action annotations out of the range of video duration
        data_prefix=dict(frames='rawframes/val'),  # TH14 val set is used for training, following command practices
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
        window_size=window_size,
        window_stride=window_stride_test,
        frame_interval=frame_interval,
        skip_short=False,
        skip_wrong=True,
        data_prefix=dict(frames='rawframes/test'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='TadMetric', merge_windows=True,
                     iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7], nms_cfg=dict(type='nms', iou_thr=0.6))
test_evaluator = val_evaluator
# efficient_conv_bn_eval = ['backbone'] # only work for slowonly
