import random
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
from mmaction.datasets.transforms import RawFrameDecode
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from my_modules.task_modules.segments_ops import segment_overlaps


@TRANSFORMS.register_module()
class LoadFeature(BaseTransform):
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if 'feat' in results:
            feat = results['feat']
        else:
            feat = np.load(results['feat_path'])
        feat_start, feat_len = results['feat_start'], results['feat_len']
        results['feat'] = feat[feat_start: feat_start + feat_len]

        return results


@TRANSFORMS.register_module()
class PadFeature(BaseTransform):

    def __init__(self,
                 pad_length: int = None,
                 pad_length_divisor: int = 1,
                 pad_value=0.0):
        assert pad_length is None or pad_length % pad_length_divisor == 0, "pad_length must be divisible by pad_size_divisor"
        self.pad_length = pad_length
        self.pad_length_divisor = pad_length_divisor
        self.pad_value = pad_value

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, feat_len = results['feat'], results['feat_len']
        assert len(feat) == feat_len

        # Case 1: Pad to specified length
        if self.pad_length is not None and feat_len < self.pad_length:
            feat = np.pad(feat, ((0, self.pad_length - feat_len), (0, 0)), constant_values=self.pad_value)
            feat_len = self.pad_length

        # Case 2 & 3: Pad to make divisible (applies when feat_len >= pad_length or only divisible is set)
        if feat_len % self.pad_length_divisor != 0:
            pad_amount = self.pad_length_divisor - (feat_len % self.pad_length_divisor)
            feat = np.pad(feat, ((0, pad_amount), (0, 0)), constant_values=self.pad_value)

        results['feat'] = feat
        return results


@TRANSFORMS.register_module()
class RandomSlice(BaseTransform):
    """Randomly slice a sub-video or sub-feature from a video/feature along the temporal axis."""

    def __init__(self,
                 window_size: int,  # the feature length input to the model
                 iof_thr: float = 0.75,
                 attempts: int = 1000,
                 size_jittering: Tuple[float] = None,
                 frame_interval: int = 1):
        self.window_size = window_size
        # Only windows with IoF (Intersection over Foreground) > iof_thr for at least one action are valid.
        self.iof_thr = iof_thr
        self.attempts = attempts
        self.size_jittering = size_jittering
        self.frame_interval = frame_interval

    @staticmethod
    def get_valid_mask(segments, patch, iof_thr, ignore_flags=None):
        gt_iofs = segment_overlaps(segments, patch, mode='iof')[:, 0]
        # patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        # iofs = np.maximum(gt_iofs, patch_iofs)
        # mask = iofs >= iof_thr
        mask = gt_iofs >= iof_thr
        if ignore_flags is not None:
            mask = mask & ~ignore_flags
        return mask

    def transform(self,
                  results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        if 'feat_len' in results:
            max_len = results['feat_len']
            # Convert the format of segment annotations from second-unit to feature-unit.
            # feat_stride tells that we extract one feature for every 'feat_stride' frames
            action_segments = results['segments'] * results['fps'] / results['feat_stride']
        elif 'total_frames' in results:
            max_len = results['total_frames']
            # Convert the format of segment annotations from second-unit to frame-unit.
            action_segments = results['segments'] * results['fps']
        else:
            raise NotImplementedError

        # Conduct random slicing
        if max_len > self.window_size:
            crop_size = self.window_size
        elif self.size_jittering is not None:
            # TODO: we follow the ActionFormer to only conduct size jittering when feature length smaller than
            #  the window size. However, theoretically, the size jittering should be a independent operation.
            crop_size = random.randint(
                max(round(self.size_jittering[0] * max_len), 1),
                min(round(self.size_jittering[1] * max_len), max_len))
        else:
            crop_size = max_len

        for i in range(self.attempts):
            start_idx = random.randint(0, max_len - crop_size)
            end_idx = start_idx + crop_size

            # If no segments in the cropped window, then re-crop. Ignored segments (Ambiguous) do not count.
            valid_mask = self.get_valid_mask(action_segments,
                                             np.array([[start_idx, end_idx]], dtype=np.float32),
                                             iof_thr=self.iof_thr,
                                             ignore_flags=results.get('gt_ignore_flags',
                                                                      np.full(action_segments.shape[0], False)))
            if not valid_mask.any():
                continue
            else:
                break
        else:
            raise RuntimeError(
                f"Could not found a valid crop after {self.attempts} attempts, "
                f"you may need modify the window size or number of attempts")

        # Convert the segment annotations to be relative to the cropped window.
        action_segments = action_segments[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
        results['segments'] = action_segments
        results['labels'] = results['labels'][valid_mask]
        if 'gt_ignore_flags' in results:
            results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_mask]

        if 'feat_len' in results:
            results['feat_start'] = start_idx
            results['feat_len'] = crop_size
        elif 'total_frames' in results:
            results['frame_inds'] = np.arange(start_idx, end_idx, step=self.frame_interval)
            results['frame_interval'] = self.frame_interval
            results['clip_len'] = crop_size // self.frame_interval

        return results


@TRANSFORMS.register_module()
class PackTADInputs(BaseTransform):
    """Pack the inputs data for the temporal action detection

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    @staticmethod
    def map_to_mmdet(results: dict) -> dict:
        """
        Modify the keys/values to be consistent with mmdet
        Args:
            results:

        Returns:

        """

        # 'segments' to 'gt_bboxes' and [x1 x2] to [x1 y1 x2 y2]
        gt_bboxes = np.insert(results['segments'], 2, 0.9, axis=-1)
        gt_bboxes = np.insert(gt_bboxes, 1, 0.1, axis=-1)
        results['gt_bboxes'] = gt_bboxes
        if 'overlap' in results and results['overlap'].size > 0:
            overlap = np.insert(results['overlap'], 2, 0.9, axis=-1)
            overlap = np.insert(overlap, 1, 0.1, axis=-1)
            results['overlap'] = overlap

        results.update({'gt_bboxes_labels': results.pop('labels')})
        results.update({"img_id": results.pop("video_name")})
        if 'feat' in results:
            # from [T, C] to [C, 1, T], mimic the [C, H, W]
            results.update({'img': results.pop('feat').T[:, np.newaxis, :]})
            results.update({'ori_shape': (1, results.pop('feat_len'))})
            results.update({'img_shape': results['ori_shape']})  # for ActionFormer
            # results.update({'img_shape': (1, results['img'].shape[-1])})  #  for DITA

        else:
            assert 'imgs' in results
            results.update({'img': results.pop('imgs')})
            results.update({'ori_shape': (1, results.pop('valid_len'))})
            results.update({'img_shape': (1, results['img'].shape[2] * results['num_clips'])})

        results['img_path'] = ''
        results['scale_factor'] = [1.0, 1.0]  # The second depends on if the backbone scale down temporal length
        results['flip'] = False
        results['flip_direction'] = None

        return results

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        results = self.map_to_mmdet(results)
        packed_results = dict()
        img = results['img']
        assert len(img.shape) == 5 or len(img.shape) == 3  # M, C, T, H, W or C, 1, T, for video frames or features
        if not img.flags.c_contiguous:
            img = np.ascontiguousarray(img)
            img = to_tensor(img)
        else:
            img = to_tensor(img).contiguous()

        packed_results['inputs'] = img

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
        ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
        instance_data['bboxes'] = to_tensor(results['gt_bboxes'][valid_idx])
        instance_data['labels'] = to_tensor(results['gt_bboxes_labels'][valid_idx])
        ignore_instance_data['bboxes'] = to_tensor(results['gt_bboxes'][ignore_idx])
        ignore_instance_data['labels'] = to_tensor(results['gt_bboxes_labels'][ignore_idx])

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # pad_shape: the shape after padding to be divisible by an instant
        # batch_input_shape: the shape after padding to be divisible by an instant and padding to the max sample shape

        data_sample.set_metainfo({
            'batch_input_shape': tuple(results['img_shape']),
            'pad_shape': tuple(results['img_shape'])
        })

        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PseudoFrameDecode(RawFrameDecode):

    def transform(self, results: Dict):
        imgs = [np.random.rand(160, 160, 3).astype(np.float32) for i in results['frame_inds']]
        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class TemporalSegment(BaseTransform):

    def __init__(self, num_clips=None, clip_len=None):
        super().__init__()
        self.num_clips = num_clips
        self.clip_len = clip_len

    def transform(self, results: Dict):
        imgs = results['imgs']
        num_imgs = len(imgs)
        if self.num_clips is not None:
            num_clips = self.num_clips
            if self.clip_len is not None:
                clip_len = self.clip_len
                assert num_clips * clip_len == num_imgs
            else:
                clip_len = num_imgs // num_clips
        else:
            assert self.clip_len is not None
            clip_len = self.clip_len
            num_clips = num_imgs // clip_len

        results['num_clips'] = num_clips
        results['clip_len'] = clip_len
        return results


@TRANSFORMS.register_module()
class Pad3D(BaseTransform):
    """Pad video frames.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_value (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_value=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_value = pad_value
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    @staticmethod
    def impad(img, shape, pad_value=0):
        """Pad an image or images to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple[int]): Expected padding shape (h, w).
            pad_value (Number | Sequence[Number]): Values to be filled in padding
                areas. Default: 0.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_value, (int, float)):
            assert len(pad_value) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + (img.shape[-1],)
        assert len(shape) == len(img.shape)
        for s, img_s in zip(shape, img.shape):
            assert s >= img_s, f"pad shape {s} should be greater than image shape {img_s}"
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_value
        pad[:img.shape[0], :img.shape[1], :img.shape[2], ...] = img
        return pad

    @staticmethod
    def impad_to_divisible(img, divisor, pad_value=0):
        """Pad an image to ensure each edge to be multiple to some number.
        Args:
            img (ndarray): Image to be padded.
            divisor (int): Padded image edges will be multiple to divisor.
            pad_value (Number | Sequence[Number]): Same as :func:`impad`.
        Returns:
            ndarray: The padded image.
        """
        pad_shape = tuple(
            int(np.ceil(shape / divisor)) * divisor for shape in img.shape[:-1])
        return Pad3D.impad(img, pad_shape, pad_value)

    def transform(self, results):
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        if self.size is not None:
            padded_imgs = self.impad(
                np.array(results['imgs']), shape=self.size, pad_value=self.pad_value)
        elif self.size_divisor is not None:
            padded_imgs = self.impad_to_divisible(
                np.array(results['imgs']), self.size_divisor, pad_value=self.pad_value)
        else:
            raise AssertionError("Either 'size' or 'size_divisor' need to be set, but both None")
        results['imgs'] = list(padded_imgs)  # change back to mmaction-style (list of) imgs
        results['pad_tsize'] = padded_imgs.shape[0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_value={self.pad_value})'
        return repr_str
