_base_ = [
    './tadtr.py'
]

# 1. Use cosine annealing lr to replace the original step lr in TadTR
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=1e-4))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10],  # 14 for TadTR
        gamma=0.1),
]
# train_cfg = dict(
#     type='EpochBasedTrainLoop', max_epochs=16, val_interval=1)
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.001,
#         by_epoch=True,
#         begin=0,
#         end=4,
#         convert_to_iter_based=True),
#     dict(
#         type='CosineAnnealingLR',
#         by_epoch=True,
#         T_max=12,
#         begin=4,
#         end=16,
#         eta_min_ratio=0.01,
#         convert_to_iter_based=True)
# ]

# 2. Use the self-supervised features (VideoMAE2)
train_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PadFeature', window_size=256),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'window_offset'))]
test_pipeline = [
    dict(type='LoadFeature'),
    dict(type='PadFeature', window_size=256),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'window_offset', 'overlap'))]
train_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 iof_thr=0.75,
                 window_stride=64,  # overlap=0.75
                 pipeline=train_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))
val_dataloader = dict(
    dataset=dict(feat_stride=4,
                 window_size=256,
                 window_stride=192,  # overlap=0.25
                 pipeline=test_pipeline,
                 data_prefix=dict(feat='features/thumos_feat_VideoMAE2-RGB_I3D-Flow_2432')))
test_dataloader = val_dataloader

# 3. Use multi-level features via temporal 1d convolution layers
# model setting
model = dict(
    type='CustomDINO',
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=False,      # must be True?
    backbone=dict(type='PseudoBackbone', multi_scale=False),  # No backbone since we use pre-extracted features.
    neck=[
        dict(
            type='DownSampler1D',
            num_levels=4,
            in_channels=2432,
            out_channels=2432,
            out_indices=(0, 1, 2, 3),
            mask=False),
        dict(
            type='ChannelMapper',
            in_channels=[2432, 2432, 2432, 2432],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4)],
    positional_encoding=dict(offset=-0.5),
    encoder=dict(num_layers=4, layer_cfg=dict(self_attn_cfg=dict(num_levels=4)), memory_fuse=True),
    decoder=dict(num_layers=4, layer_cfg=dict(cross_attn_cfg=dict(num_levels=4))),
    bbox_head=dict(type='CustomDINOHead', num_classes=20, sync_cls_avg_factor=True,
                   loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),  # 2.0
                   loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                   loss_iou=dict(_delete_=True, type='GIoU1dLoss', loss_weight=2.0)),
    dn_cfg=dict(label_noise_scale=0.5, box_noise_scale=1.0,
                group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(assigner=dict(type='HungarianAssigner',
                                 match_costs=[dict(type='FocalLossCost', weight=2.0, gamma=2.0, alpha=0.25),
                                              dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                                              dict(type='IoU1dCost', iou_mode='giou', weight=2.0)]),

                   )

)
val_evaluator = dict(
    type='TH14Metric',
    metric='mAP',
    iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7],
    nms_cfg=dict(type='nms', iou_thr=0.4),
    max_per_video=False,
    score_thr=0.0,  # screen out the detections with score less than score_thr
    duration_thr=0.0)   # screen out the detections with duration less than duration_thr (in second)
test_evaluator = val_evaluator
