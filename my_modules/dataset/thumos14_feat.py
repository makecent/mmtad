import warnings
from copy import deepcopy
import os.path as osp

import mmengine
import numpy as np
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from mmengine import print_log, MMLogger

from my_modules.dataset.transform import SlidingWindow


@DATASETS.register_module()
class Thumos14FeatDataset(BaseDetDataset):
    """Thumos14 dataset for temporal action detection.
    fix_slice=False, on_the_fly=True:   Slice the feature windows randomly.
    fix_slice=True, on_the_fly=False:   Slice the feature windows with fixed stride. All feat windows are pre-loaded (Fast but high memory).
    fix_slice=True, on_the_fly=True:    Slice the feature windows with fixed stride. Feat windows are loaded as needed.
    fix_slice=False, on_the_fly=False:  Not applicable. Dataset of different epoch should have different random slices.
    """

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                             'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'),
                    wrong_videos=('video_test_0000270', 'video_test_0001292', 'video_test_0001496'))

    def __init__(self,
                 feat_stride,  # feature are extracted every n frames
                 skip_short=False,  # skip too short annotations
                 skip_wrong=False,  # skip videos that are wrong annotated
                 fix_slice=True,
                 # whether slice the feature to windows in advance with fixed stride, or slice randomly.
                 on_the_fly=False,  # whether slice the feature on the fly and pre-slice.
                 tadtr_style=True,  # whether slice the feature in a TadTR style.
                 iof_thr=0.75,  # The Intersection over Foreground (IoF) threshold used to filter sliding windows.
                 window_size=None,  # the window size of sliding window.
                 window_stride=None,  # the window stride of sliding window.
                 **kwargs):
        self.feat_stride = feat_stride
        self.skip_short = skip_short
        self.skip_wrong = skip_wrong
        self.fix_slice = fix_slice
        self.on_the_fly = on_the_fly
        self.tadtr_style = tadtr_style
        self.iof_thr = iof_thr
        self.window_size = window_size
        self.window_stride = window_stride
        if fix_slice:
            assert isinstance(window_size, int)
            assert isinstance(window_stride, int)
        super(Thumos14FeatDataset, self).__init__(**kwargs)

    def load_data_list(self):
        # feat_offset = 0.5 * 16 / self.feat_stride
        data_list = []
        ann_file = mmengine.load(self.ann_file)
        for video_name, video_info in ann_file.items():
            if self.skip_wrong and video_name in self.metainfo['wrong_videos']:
                continue
            # Parsing ground truth
            segments, labels, ignore_flags = self.parse_labels(video_name, video_info)

            # Loading features
            feat_path = osp.join(self.data_prefix['feat'], video_name) + '.npy'
            if mmengine.exists(feat_path):
                feat = np.load(feat_path)
            else:
                warnings.warn(f"Cannot find feature file {feat_path}, skipped")
                continue
            feat_len = len(feat)

            data_info = dict(video_name=video_name,
                             duration=float(video_info['duration']),
                             fps=float(video_info['FPS']),
                             feat_stride=self.feat_stride,
                             segments=segments,
                             labels=labels,
                             gt_ignore_flags=ignore_flags)

            if not self.fix_slice and self.on_the_fly:  # slice randomly
                assert isinstance(self.pipeline.transforms[0], SlidingWindow)
                data_info.update(dict(feat=feat, feat_len=feat_len))
                data_list.append(data_info)
            else:
                # Perform fixed-stride sliding window
                if self.tadtr_style:
                    # TadTR handles all the complete windows, and if applicable, plus one tail window that covers the remaining content
                    num_complete_windows = max(0, feat_len - self.window_size) // self.window_stride + 1
                    # feat_windows = feat_windows[:num_complete_windows]
                    start_indices = np.arange(num_complete_windows) * self.window_stride
                    end_indices = (start_indices + self.window_size).clip(max=feat_len)
                    # Handle the tail window
                    if (feat_len - self.window_size) % self.window_stride != 0 and feat_len > self.window_size:
                        tail_start = self.window_stride * num_complete_windows if self.test_mode else feat_len - self.window_size
                        start_indices = np.append(start_indices, tail_start)
                        end_indices = np.append(end_indices, feat_len)
                else:
                    start_indices = np.arange(feat_len // self.window_stride + 1) * self.window_stride
                    end_indices = (start_indices + self.window_size).clip(max=feat_len)
                # Compute overlapped regions
                overlapped_regions = np.array(
                    [[start_indices[i], end_indices[i - 1]] for i in range(1, len(start_indices))])
                overlapped_regions = overlapped_regions * self.feat_stride / data_info['fps']

                for start_idx, end_idx in zip(start_indices, end_indices):
                    assert start_idx < end_idx <= feat_len, f"invalid {start_idx, end_idx, feat_len}"
                    feat_window = feat[start_idx: end_idx]
                    feat_win_len = len(feat_window)

                    # Padding windows that are shorter than the target window size.
                    if feat_win_len < self.window_size:
                        feat_window = np.pad(feat_window,
                                             ((0, self.window_size - feat_win_len), (0, 0)),
                                             constant_values=0)
                    data_info.update(dict(offset=start_idx,
                                          feat_len=feat_win_len))  # before padding for computing the valid feature mask
                    if self.on_the_fly:
                        assert isinstance(self.pipeline.transforms[0], SlidingWindow)
                        data_info.update(dict(feat_path=feat_path))
                    else:
                        data_info.update(dict(feat=feat_window))

                    # Convert the format of segment annotations from second-unit to feature-unit.
                    segments_f = segments * data_info['fps'] / self.feat_stride
                    if self.test_mode:
                        data_info.update(dict(overlap=overlapped_regions))
                    else:
                        # During the training, windows has no segment annotated are skipped
                        # Also known as Integrity-based instance filtering (IBIF)
                        valid_mask = SlidingWindow.get_valid_mask(segments_f,
                                                                  np.array([[start_idx, end_idx]], dtype=np.float32),
                                                                  iof_thr=self.iof_thr,
                                                                  ignore_flags=ignore_flags)
                        if not valid_mask.any():
                            continue
                        # Convert the segment annotations to be relative to the feature window
                        segments_f = segments_f[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
                        labels_f = labels[valid_mask]
                        ignore_flags_f = ignore_flags[valid_mask]
                        data_info.update(dict(segments=segments_f,
                                              labels=labels_f,
                                              gt_ignore_flags=ignore_flags_f))

                    data_list.append(deepcopy(data_info))
        print_log(f"number of feature windows:\t {len(data_list)}", logger=MMLogger.get_current_instance())
        # if self.test_mode:
        #     segments = []
        #     for i in data_list:
        #         segments.append(i['segments'][:, 1] - i['segments'][:, 0])
        #
        #     seg_len = np.concatenate(segments)
        #     from mmengine import dump
        #     dump(seg_len, 'test_seg_len.pkl')
        # else:
        #     pass
        #     # dump(seg_len, 'train_seg_len.pkl')
        return data_list

    def parse_labels(self, video_name, video_info):
        # Segments information
        segments = []
        labels = []
        ignore_flags = []
        video_duration = video_info['duration']
        for segment, label in zip(video_info['segments'], video_info['labels']):

            # Skip annotations that are out of range.
            if not (0 <= segment[0] < segment[1] <= video_duration) and self.skip_wrong:
                print_log(f"invalid segment annotation in {video_name}: {segment}, duration={video_duration}, skipped",
                          logger=MMLogger.get_current_instance())
                continue

            # Skip annotations that are too short. The threshold could be the stride of feature extraction.
            # For example, if the features were extracted every 8 frames,
            # then the threshold should be greater than 8/30 = 0.27s
            if isinstance(self.skip_short, (int, float)):
                if segment[1] - segment[0] < self.skip_short:
                    print_log(f"too short segment annotation in {video_name}: {segment}, skipped",
                              logger=MMLogger.get_current_instance())

            # Ambiguous annotations are labeled as ignored ground truth
            if label == 'Ambiguous':
                labels.append(-1)
                ignore_flags.append(True)
            else:
                labels.append(self.metainfo['classes'].index(label))
                ignore_flags.append(False)
            segments.append(segment)

        return np.array(segments, np.float32), np.array(labels, np.int64), np.array(ignore_flags, dtype=bool)
