_base_ = [
    'default_runtime.py'
]

max_seq_len = 2304
# model setting
model = dict(type='SingleStageDetector',
             data_preprocessor=dict(type='DetDataPreprocessor'),
             backbone=dict(type='PseudoBackbone', multi_scale=False),
             neck=dict(
                 type='convTransformer',
                 arch=(2, 2, 5),
                 attn_window_size=[19, 19, 19, 19, 19, 19],
                 in_channels=2048,  # the dimension of I3D features
                 embed_dims=512,
                 num_heads=4,
                 max_seq_len=max_seq_len,
                 path_pdrop=0.1),
             bbox_head=dict(
                 type='ActionFormerHead',
                 num_classes=200,
                 in_channels=512,
                 stacked_convs=2,
                 feat_channels=512,
                 strides=(1, 2, 4, 8, 16, 32),
                 # strides of the head input features with respect to the model input feature
                 regress_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0, reduction='none'),
                 loss_bbox=dict(type='DIoU1dLoss', loss_weight=1.0, reduction='none'),
                 prior_prob=0.01,
                 init_loss_norm=100.0),
             test_cfg=dict(
                 score_thr=0.001,  # score threshold before NMS
                 nms_pre=2000,  # (2000, 5000) are used for different configs
                 min_bbox_size=[0.05, 0],  # (0.001, 0.05) are used in different configs
                 # min_bbox_size (w, h) before NMS. As we hack T as W, we only set the width threshold.
                 nms=dict(
                     type='soft_nms',
                     iou_threshold=0.1,
                     sigma=0.5,  # (0.4, 0.5, 0.75, 0.9) are used in different configs
                     min_score=0.001),   # score threshold after NMS
                 with_score_voting=False,  # only triggered when multi-class is False, i.e., proposal generation.
                 max_per_img=200
             )  # (100, 200, 1000, 2000) are used in different configs
)

# optimizer
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

# dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

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

val_evaluator = dict(type='TadMetric', iou_thrs=[0.3, 0.4, 0.5, 0.6, 0.7])
test_evaluator = val_evaluator
