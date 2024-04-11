# dataset settings
dataset_type = 'THUMOS14Dataset'
data_root = 'my_data/thumos14/'

frame_interval = 8
window_size = 960
window_stride_train = 720  # overlap=0.75
window_stride_test = 720  # overlap=0.25
num_clips = 15        # 192/12=16 frame per clip
# window_size = 480
# window_stride_train = 360  # overlap=0.75
# window_stride_test = 360  # overlap=0.25
# num_clips = 6        # 96/6=16 frame per clip
# img_shape = (112, 112)
# img_shape_test = (128, 128)
img_shape = (224, 224)
img_shape_test = (224, 224)


train_pipeline = [
    # dict(type='PseudoFrameDecode'),
    dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=img_shape, keep_ratio=False),
    # dict(type='mmaction.Resize', scale=(128, -1), keep_ratio=True), # scale images' short-side to 128, keep aspect ratio
    # dict(type='mmaction.RandomCrop', size=img_shape[0]),
    dict(type='mmaction.Flip', flip_ratio=0.5),
    dict(type='Pad3d', size=(window_size//frame_interval, *img_shape)),
    dict(type='TemporalSegment', num_clips=num_clips),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'frame_interval', 'window_offset'))
]
test_pipeline = [
    # dict(type='PseudoFrameDecode'),
    dict(type='mmaction.RawFrameDecode'),
    dict(type='mmaction.Resize', scale=img_shape_test, keep_ratio=False),
    # dict(type='mmaction.Resize', scale=(128, -1), keep_ratio=True),
    # dict(type='mmaction.CenterCrop', crop_size=img_shape_test),
    dict(type='Pad3d', size=(window_size//frame_interval, *img_shape_test)),
    dict(type='TemporalSegment', num_clips=num_clips),
    dict(type='mmaction.FormatShape', input_format='NCTHW'),
    dict(type='PackTADInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction',
                    'fps', 'frame_interval', 'window_offset', 'overlap'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/mmtad/thumos14_val.json',
        fix_slice=True,
        window_size=window_size,
        window_stride=window_stride_train,
        frame_interval=frame_interval,
        iof_thr=0.75,
        skip_short=0.3,  # skip action annotations with duration less than 0.3 seconds
        skip_wrong=True,  # skip action annotations out of the range of video duration
        data_prefix=dict(frames='rawframes/val'),   # TH14 val set is used for training, following command practices
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
        window_size=window_size,
        window_stride=window_stride_test,
        frame_interval=frame_interval,
        skip_short=False,
        skip_wrong=True,
        data_prefix=dict(frames='rawframes/test'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader
