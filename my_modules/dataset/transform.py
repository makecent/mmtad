import random
from typing import Dict, Optional, Tuple, List, Union

import numpy as np
import torch
from mmaction.datasets.transforms import RawFrameDecode
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6,
                     detect_overlap_edge=False):
    """Calculate overlap between two set of segments.
    If ``is_aligned`` is ``False``, then calculate the ious between each
    segment of segments1 and segments2, otherwise the ious between each aligned
     pair of segments1 and segments2.
    Args:
        segments1 (Tensor): shape (m, 2) in <t1, t2> format or empty.
        segments2 (Tensor): shape (n, 2) in <t1, t2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    Example:
        >>> segments1 = torch.FloatTensor([
        >>>     [0, 10],
        >>>     [10, 20],
        >>>     [32, 38],
        >>> ])
        >>> segments2 = torch.FloatTensor([
        >>>     [0, 20],
        >>>     [0, 19],
        >>>     [10, 20],
        >>> ])
        >>> segment_overlaps(segments1, segments2)
        tensor([[0.5000, 0.5263, 0.0000],
                [0.0000, 0.4500, 1.0000],
                [0.0000, 0.0000, 0.0000]])
    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 9],
        >>> ])
        >>> assert tuple(segment_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(segment_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(segment_overlaps(empty, empty).shape) == (0, 0)
    """

    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    segments1, segments2 = segments1.float(), segments2.float()

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments' last dimenstion is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        ious = segments1.new(rows, 1) if is_aligned else segments2.new(rows, cols)
        return ious.numpy() if is_numpy else ious

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                 0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious


@TRANSFORMS.register_module()
class SlidingWindow(BaseTransform):

    def __init__(self,
                 window_size: int,  # the feature length input to the model
                 iof_thr=0.75,
                 attempts=1000,
                 just_loading=False,
                 crop_ratio=None):
        self.window_size = window_size
        # Only windows with IoF (Intersection over Foreground) > iof_thr for at least one action are valid.
        self.iof_thr = iof_thr
        self.attempts = attempts
        self.crop_ratio = crop_ratio
        self.just_loading = just_loading  # If True, sliding window was completed, we just load the features here.

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
        if not self.just_loading:
            feat, feat_len = results['feat'], results['feat_len']
            # Convert the format of segment annotations from second-unit to feature-unit.
            # feat_stride tells that we extract one feature for every 'feat_stride' frames
            segments_feat = results['segments'] * results['fps'] / results['feat_stride']

            # Conduct sliding window
            if feat_len > self.window_size:
                crop_size = self.window_size
            elif self.crop_ratio is not None:
                crop_size = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len))
            else:
                crop_size = feat_len

            for i in range(self.attempts):
                start_idx = random.randint(0, feat_len - crop_size)
                end_idx = start_idx + crop_size

                # If no segments in the cropped window, then re-crop. Ignored segments (Ambiguous) do not count.
                valid_mask = self.get_valid_mask(segments_feat,
                                                 np.array([[start_idx, end_idx]], dtype=np.float32),
                                                 iof_thr=self.iof_thr,
                                                 ignore_flags=results.get('gt_ignore_flags',
                                                                          np.full(segments_feat.shape[0], False)))
                if not valid_mask.any():
                    continue

                # Convert the segment annotations to be relative to the cropped window.
                segments_feat = segments_feat[valid_mask].clip(min=start_idx, max=end_idx) - start_idx
                results['segments'] = segments_feat
                results['labels'] = results['labels'][valid_mask]
                if 'gt_ignore_flags' in results:
                    results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_mask]
                results['feat'] = feat[start_idx: end_idx]
                results['feat_len'] = crop_size
                break
            else:
                raise RuntimeError(
                    f"Could not found a valid crop after {self.attempts} attempts, "
                    f"you may need modify the window size or number of attempts")
        else:
            window_offset, feat_len, feat_path = results['window_offset'], results['feat_len'], results['feat_path']
            feat = np.load(feat_path)[window_offset: window_offset + feat_len]
            crop_size = feat_len

        if crop_size < self.window_size:
            # Padding the feature to window size if the feat is too short
            feat = np.pad(feat, ((0, self.window_size - feat_len), (0, 0)), constant_values=0)
        results['feat'] = feat

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
            results.update({'img': results.pop('feat')[None]})
            # results.update({'img_shape': results['ori_shape']})
            results.update({'img_shape': (1, results['img'].shape[1])})
            results.update({'ori_shape': (1, results.pop('feat_len'))})

        else:
            assert 'imgs' in results
            results.update({'img': results.pop('imgs')})
            results.update({'img_shape': (1, results['img'].shape[2])})
            results.update({'ori_shape': (1, results.pop('valid_len'))})

        results['img_path'] = ''
        results['scale_factor'] = [1.0, 1.0]
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
        assert len(img.shape) == 5
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
