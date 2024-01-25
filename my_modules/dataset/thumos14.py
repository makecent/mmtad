# adapted from basicTAD
import re
import warnings
from copy import deepcopy
from pathlib import Path

import mmengine
import numpy as np
from mmdet.datasets import BaseDetDataset
from mmdet.registry import DATASETS
from mmengine import print_log, MMLogger
import os.path as osp

from my_modules.dataset.transform import SlidingWindow


def make_regex_pattern(fixed_pattern):
    # Use regular expression to extract number of digits
    num_digits = re.search(r'\{:(\d+)\}', fixed_pattern).group(1)
    # Build the pattern string using the extracted number of digits
    pattern = fixed_pattern.replace('{:' + num_digits + '}', r'\d{' + num_digits + '}')
    return pattern


@DATASETS.register_module()
class THUMOS14Dataset(BaseDetDataset):
    """THUMOS14 dataset for temporal action detection."""

    metainfo = dict(classes=('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                             'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                             'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                             'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking'),
                    wrong_videos=('video_test_0000270', 'video_test_0001292', 'video_test_0001496'))

    def __init__(self,
                 window_size,
                 frame_interval,
                 window_stride=None,
                 skip_short=False,  # skip too short annotations
                 skip_wrong=False,  # skip videos that are wrong annotated
                 fix_slice=True,
                 tad_style=True,
                 # whether slice the feature to windows in advance with fixed stride, or slice randomly.
                 iof_thr=0.75,  # The Intersection over Foreground (IoF) threshold used to filter sliding windows.
                 start_index=0,
                 filename_tmpl='img_{:05}.jpg',
                 modality: str = 'RGB',
                 **kwargs):
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.window_stride = window_stride
        self.skip_short = skip_short
        self.skip_wrong = skip_wrong
        self.fix_slice = fix_slice
        self.tad_style = tad_style
        self.iof_thr = iof_thr
        self.start_index = start_index
        self.filename_tmpl = filename_tmpl
        self.modality = modality

        super(THUMOS14Dataset, self).__init__(**kwargs)

    def load_data_list(self):
        data_list = []
        ann_file = mmengine.load(self.ann_file)
        for video_name, video_info in ann_file.items():
            if self.skip_wrong and video_name in self.metainfo['wrong_videos']:
                continue

            # Parsing ground truth
            segments, labels, ignore_flags = self.parse_labels(video_name, video_info)

            frame_dir = Path(self.data_prefix['frames']).joinpath(video_name)
            # if not frame_dir.exists():
            #     warnings.warn(f'{frame_dir} does not exist.')
            #     continue

            data_info = dict(video_name=video_name,
                             frame_dir=osp.join(self.data_prefix['frames'], video_name),
                             duration=float(video_info['duration']),
                             fps=float(video_info['FPS']),
                             frame_interval=self.frame_interval,
                             segments=segments,
                             labels=labels,
                             gt_ignore_flags=ignore_flags)

            # Get the number of frames of the video
            # pattern = make_regex_pattern(self.filename_tmpl)
            # imgfiles = [img for img in frame_dir.iterdir() if re.fullmatch(pattern, img.name)]
            # total_frames = len(imgfiles)
            total_frames = video_info['num_frame']

            if not self.fix_slice:  # slice randomly
                assert isinstance(self.pipeline.transforms[0], SlidingWindow)
                data_info.update(dict(total_frames=total_frames))
                data_list.append(data_info)
            else:
                if self.tad_style:
                    # TadTR handles all the complete windows, and if applicable, plus one tail window that covers the remaining content
                    num_complete_windows = max(0, total_frames - self.window_size) // self.window_stride + 1
                    # feat_windows = feat_windows[:num_complete_windows]
                    start_indices = np.arange(num_complete_windows) * self.window_stride
                    end_indices = (start_indices + self.window_size).clip(max=total_frames)
                    # Handle the tail window
                    if (total_frames - self.window_size) % self.window_stride != 0 and total_frames > self.window_size:
                        tail_start = self.window_stride * num_complete_windows if self.test_mode else total_frames - self.window_size
                        start_indices = np.append(start_indices, tail_start)
                        end_indices = np.append(end_indices, total_frames)
                else:
                    # Sliding window with fixed stride, and the last windows may be incomplete and need padding
                    start_indices = np.arange(0, total_frames, self.window_size)
                    end_indices = (start_indices + self.window_size).clip(max=total_frames)

                # Compute overlapped regions
                overlapped_regions = np.array(
                    [[start_indices[i], end_indices[i - 1]] for i in range(1, len(start_indices))])
                overlapped_regions = overlapped_regions * self.window_stride / data_info['fps']

                for start_idx, end_idx in zip(start_indices, end_indices):
                    data_info.update(dict(
                        window_offset=start_idx,
                        valid_len=end_idx - start_idx,
                        frame_inds=np.arange(start_idx, end_idx, self.frame_interval)))
                    assert start_idx < end_idx <= total_frames, f"invalid {start_idx, end_idx, total_frames}"
                    if self.test_mode:
                        data_info.update(dict(overlap=overlapped_regions))
                    else:
                        # During the training, windows have low action footage are skipped
                        # Also known as Integrity-based instance filtering (IBIF)
                        segments_f = segments * video_info['FPS']
                        valid_mask = SlidingWindow.get_valid_mask(segments_f,
                                                                  np.array([[start_idx, end_idx]],
                                                                           dtype=np.float32),
                                                                  iof_thr=self.iof_thr,
                                                                  ignore_flags=ignore_flags)
                        if not valid_mask.any():
                            continue
                        # Convert the segment annotations to be relative to the feature window
                        segments_f = segments_f[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
                        labels_f = labels[valid_mask]
                        ignore_flags_f = ignore_flags[valid_mask]
                        data_info.update(dict(
                            segments=segments_f,
                            labels=labels_f,
                            gt_ignore_flags=ignore_flags_f))

                    data_list.append(deepcopy(data_info))
        assert len(data_list) > 0
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
                    continue

            # Ambiguous annotations are labeled as ignored ground truth
            if label == 'Ambiguous':
                labels.append(-1)
                ignore_flags.append(True)
            else:
                labels.append(self.metainfo['classes'].index(label))
                ignore_flags.append(False)
            segments.append(segment)
        return np.array(segments, np.float32), np.array(labels, np.int64), np.array(ignore_flags, dtype=bool)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['start_index'] = self.start_index
        data_info['modality'] = self.modality
        data_info['filename_tmpl'] = self.filename_tmpl
        data_info['num_clips'] = 1   # just for compatible with mmaction2
        data_info['clip_len'] = self.window_size // self.frame_interval

        return data_info
