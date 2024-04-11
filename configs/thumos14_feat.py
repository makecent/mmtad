# dataset settings
dataset_type = 'Thumos14FeatDataset'
data_root = 'my_data/thumos14/'

train_pipeline = [
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'feat_stride', 'window_offset'))
]
test_pipeline = [
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
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/mmtad/thumos14_val.json',
        feat_stride=8,
        window_size=128,
        window_stride=32,  # overlap=0.75
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
        ann_file='annotations/mmtad/thumos14_test.json',
        feat_stride=8,
        window_size=128,
        window_stride=96,  # overlap=0.25
        skip_short=False,
        skip_wrong=True,
        data_prefix=dict(feat='features/thumos_feat_TadTR_64input_8stride_2048'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
