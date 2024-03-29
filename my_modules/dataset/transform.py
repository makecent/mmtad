import random
from typing import Dict, Optional, Tuple, List, Union

import mmcv
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import PhotoMetricDistortion
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from my_modules.task_modules.segments_ops import segment_overlaps, convert_1d_to_2d_bboxes


@TRANSFORMS.register_module()
class LoadFeature(BaseTransform):
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if 'feat' in results:
            feat = results['feat']
        else:
            feat = np.load(results['feat_path'])
        feat_start, valid_len = results['feat_start'], results['valid_len']
        results['feat'] = feat[feat_start: feat_start + valid_len]

        return results


@TRANSFORMS.register_module()
class PadFeature(BaseTransform):

    def __init__(self,
                 pad_length: int = None,
                 pad_length_divisor: int = 1,
                 pad_value=0.0):
        assert pad_length is None or pad_length % pad_length_divisor == 0, \
            "pad_length must be divisible by pad_size_divisor"
        self.pad_length = pad_length
        self.pad_length_divisor = pad_length_divisor
        self.pad_value = pad_value

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        feat, valid_len = results['feat'], results['valid_len']
        assert len(feat) == valid_len

        # Case 1: Pad to specified length
        if self.pad_length is not None and valid_len < self.pad_length:
            feat = np.pad(feat, ((0, self.pad_length - valid_len), (0, 0)), constant_values=self.pad_value)

        # Case 2 & 3: Pad to make divisible (applies when len(feat) >= pad_length or only divisible is set)
        if len(feat) % self.pad_length_divisor != 0:
            pad_amount = self.pad_length_divisor - (len(feat) % self.pad_length_divisor)
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
        patch_iofs = segment_overlaps(patch, segments, mode='iof')[0, :]
        iofs = np.maximum(gt_iofs, patch_iofs)
        mask = iofs >= iof_thr
        # mask = gt_iofs >= iof_thr
        if ignore_flags is not None:
            mask = mask & ~ignore_flags
        return mask

    def transform(self,
                  results: Dict, ) -> Optional[Union[Dict, Tuple[List, List]]]:
        if 'valid_len' in results:
            max_len = results['valid_len']
        elif 'total_frames' in results:
            max_len = results['total_frames']
        else:
            raise NotImplementedError

        action_segments = results['segments']

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
                f"you may need increase the window size or number of attempts."
                f"Trying to find a window of size {self.window_size} in a video of size {max_len}, "
                f"which has actions {action_segments}.")

        # Convert the segment annotations to be relative to the cropped window.
        action_segments = action_segments[valid_mask].clip(min=start_idx, max=end_idx) - start_idx

        if 'valid_len' in results:
            results['feat_start'] = start_idx
            results['valid_len'] = crop_size
            results['segments'] = action_segments
        elif 'total_frames' in results:
            results['frame_inds'] = np.arange(start_idx, end_idx, step=self.frame_interval)
            results['frame_interval'] = self.frame_interval
            results['valid_len'] = len(results['frame_inds'])
            results['clip_len'] = self.window_size // self.frame_interval
            results['segments'] = action_segments / self.frame_interval
        else:
            raise NotImplementedError

        results['labels'] = results['labels'][valid_mask]
        if 'ignore_flags' in results:
            results['ignore_flags'] = results['ignore_flags'][valid_mask]

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
                 meta_keys=('video_name', 'valid_len')):
        self.meta_keys = meta_keys

    @staticmethod
    def map_to_mmdet(packed_results: dict, results: dict) -> dict:
        inputs, data_sample = packed_results['inputs'], packed_results['data_samples']
        # reformat input data and shapes
        if inputs.ndim == 2:
            # from [T, C] to [C, 1, T], mimic the [C, H, W]
            inputs_chw = inputs.unsqueeze(0).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = inputs_chw
            temporal_len = inputs_chw.size(2)
            # data_sample.set_metainfo(dict(img_shape=(1, inputs_chw.size(-1))))  # for DITA
        else:
            assert inputs.ndim == 5   # MCTHW   M = num_clips x num_crops
            temporal_len = inputs.size(2) * results['num_clips']
        # %% reformat shape to (1, T) from (H, W) for temporal detection, these values could be used for
        # computing padding mask or clipping the geographical range to detection results, etc.
        data_sample.set_metainfo(dict(ori_shape=(1, results['valid_len'])))
        data_sample.set_metainfo(dict(img_shape=(1, results['valid_len'])))
        data_sample.set_metainfo(dict(batch_input_shape=(1, temporal_len)))
        data_sample.set_metainfo(dict(pad_shape=(1, temporal_len)))
        # %% add some keys that may be used
        data_sample.set_metainfo(dict(scale_factor=(1.0, 1.0)))
        data_sample.set_metainfo(dict(flip=False))
        data_sample.set_metainfo(dict(flip_direction=None))
        # %% reformat gt_instances from (x1, x2) to (x1, y1=0.1, x2, y2=0.9)
        data_sample.gt_instances.bboxes = convert_1d_to_2d_bboxes(data_sample.gt_instances.segments)
        data_sample.ignored_instances.bboxes = convert_1d_to_2d_bboxes(data_sample.ignored_instances.segments)
        # %% reformat overlaps
        if 'overlap' in data_sample and data_sample.overlap.size > 0:
            overlap_2d = np.insert(data_sample.pop('overlap'), 2, 0.9, axis=-1)
            overlap_2d = np.insert(overlap_2d, 1, 0.1, axis=-1)
            data_sample.set_metainfo(dict(overlap=overlap_2d))
        # # %% reformat video_name to img_id
        # if 'video_name' in data_sample:
        #     data_sample.set_metainfo(dict(img_id=data_sample.video_name))

        return packed_results

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
        packed_results = dict()
        data = results['feat'] if 'feat' in results else results['imgs']
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
            data = to_tensor(data)
        else:
            data = to_tensor(data).contiguous()

        packed_results['inputs'] = data

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        valid_idx = np.where(results['ignore_flags'] == 0)[0]
        ignore_idx = np.where(results['ignore_flags'] == 1)[0]
        instance_data['segments'] = to_tensor(results['segments'][valid_idx])
        instance_data['labels'] = to_tensor(results['labels'][valid_idx])
        ignore_instance_data['segments'] = to_tensor(results['segments'][ignore_idx])
        ignore_instance_data['labels'] = to_tensor(results['labels'][ignore_idx])

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

        packed_results['data_samples'] = data_sample
        packed_results = self.map_to_mmdet(packed_results, results)
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PseudoFrameDecode(BaseTransform):

    def __init__(self, size=(224, 224)):
        super().__init__()
        self.size = size

    def transform(self, results: Dict):
        imgs = [np.random.rand(*self.size, 3).astype(np.float32) for i in results['frame_inds']]
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
class Pad3d(BaseTransform):
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
        return Pad3d.impad(img, pad_shape, pad_value)

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
        results['pad_len'] = padded_imgs.shape[0]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_value={self.pad_value})'
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortion3d(PhotoMetricDistortion):

    def __init__(self, prob=0.5, *args, **kwargs):
        # prob is the probability that the entire process is skipped
        self.prob = prob
        super().__init__(*args, **kwargs)

    def transform(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if random.uniform(0, 1) <= self.prob:
            imgs = np.array(results['imgs']).astype(np.float32)

            (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
             swap_flag, delta_value, alpha_value, saturation_value, hue_value,
             swap_value) = self._random_flags()

            # random brightness
            if brightness_flag:
                imgs += delta_value

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            if mode == 1:
                if contrast_flag:
                    imgs *= alpha_value

            # convert color from BGR to HSV
            imgs = np.array([mmcv.image.bgr2hsv(img) for img in imgs])

            # random saturation
            if saturation_flag:
                imgs[..., 1] *= saturation_value
                # For image(type=float32), after convert bgr to hsv by opencv,
                # valid saturation value range is [0, 1]
                if saturation_value > 1:
                    imgs[..., 1] = imgs[..., 1].clip(0, 1)

            # random hue
            if hue_flag:
                imgs[..., 0] += hue_value
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([mmcv.image.hsv2bgr(img) for img in imgs])

            # random contrast
            if mode == 0:
                if contrast_flag:
                    imgs *= alpha_value

            # randomly swap channels
            if swap_flag:
                imgs = imgs[..., swap_value]

            results['imgs'] = list(imgs)  # change back to mmaction-style (list of) imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate3d(BaseTransform):
    """Spatially rotate images.

    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect_101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(self,
                 limit,
                 interpolation='bilinear',
                 border_mode='constant',
                 border_value=0,
                 prob=0.5):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.prob = prob

    def transform(self, results):
        """Call function to random rotate images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Spatially rotated results.
        """

        if random.uniform(0, 1) <= self.prob:
            angle = random.uniform(*self.limit)
            imgs = [
                mmcv.image.imrotate(
                    img,
                    angle=angle,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    border_value=self.border_value) for img in results['imgs']]

            results['imgs'] = [np.ascontiguousarray(img) for img in imgs]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(limit={self.limit},'
        repr_str += f'interpolation={self.interpolation},'
        repr_str += f'border_mode={self.border_mode},'
        repr_str += f'border_value={self.border_value},'
        repr_str += f'p={self.prob})'

        return repr_str
