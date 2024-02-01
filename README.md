# The official repo for DITA
The code is not completed and the readme.md need to be updated.


# Prepare the environment
```terminal
conda create -n mmengine python=3.8 -y
conda activate mmengine
# the version of pytorch is flexible (>=2.0 is recommended), you may install pytorch depending on your machine.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim fvcore future tensorboard pytorchvideo
mim install mmengine mmdet mmaction2
```
Add the root directory to the Python path, otherwise you need add `PYTHONPATH=$PWD:$PYTHONPATH` before every run:
```terminal
cd DITA
export PYTHONPATH=$PWD:$PYTHONPATH
```
Note that once you close the terminal, you need re-run the above command as it is a temporary setting.

Running commands you need to know (refer to [openmim](https://github.com/open-mmlab/mim) for more details):

Training command:
```terminal
mim train mmaction $CONFIG --gpus $NUM_GPUS
```
Test command:
```terminal
mim test mmaction $CONFIG --gpus $NUM_GPUS --checkpoint $PATH_TO_CHECKPOINT
```
Notes:
- When $NUM_GPUS > 1, distributed training or testing will be used. You may add `--launcher pytorch` to 
use PyTorch launcher, or `--launcher slurm` to use Slurm launcher.
- The final batch_size is $NUM_GPUS * $CFG.train_dataloader.batch_size. You may need override some options when using 
different number of GPUs. For example, if you want to use 8 GPUs, you may add `--cfg-options train_dataloader.batch_size=xxx`
to reduce the batch_size on single GPU by 8 in order to keep the final batch size unchanged.
- 
# Reproduce TadTR
```terminal
mmaction2  1.2.0      https://github.com/open-mmlab/mmaction2
mmcv       2.1.0      https://github.com/open-mmlab/mmcv
mmdet      3.3.0      https://github.com/open-mmlab/mmdetection
mmengine   0.10.3     https://github.com/open-mmlab/mmengine
```
## THUMOS14 with I3D features
### Prepare data
Download the pre-extracted features from the [official repository](https://github.com/happyharrycn/actionformer_release/)
and put them in `my_data/thumos14/features/thumos_feat_TadTR_64input_8stride_2048`. We use annotation files created by ourselves.

### Train (including validation)
Train (2 GPUs as an example):
```terminal
mim train mmaction configs/repo_actionformer_th14.py --gpus 2 --launcher pytorch --cfg-options train_dataloader.batch_size=1
```
### Test
Test (2 GPUs as an example):
```terminal
mim train mmaction configs/repo_actionformer_th14.py --gpus 2 --launcher pytorch --checkpoint work_dirs/repo_actionformer_th14/latest.pth --cfg-options train_dataloader.batch_size=1
```

# Reproduce ActionFormer
```terminal
mmaction2  1.2.0      https://github.com/open-mmlab/mmaction2
mmcv       2.1.0      https://github.com/open-mmlab/mmcv
mmdet      3.3.0      https://github.com/open-mmlab/mmdetection
mmengine   0.10.3     https://github.com/open-mmlab/mmengine
```
## THUMOS14 with I3D features
### Prepare data
Download the pre-extracted features from the [official repository](https://github.com/happyharrycn/actionformer_release/)
and put them in `my_data/thumos14/features/thumos_feat_ActionFormer_16input_4stride_2048/i3d_features`. We use annotation files created by ourselves.

### Train (including validation)
Train (2 GPUs as an example):
```terminal
mim train mmaction configs/repo_actionformer_th14.py --gpus 2 --launcher pytorch --cfg-options train_dataloader.batch_size=1
```
We change batch_size to 1 (which is 2 by default) here as two GPUs are used for training. The final batch_size is still 2, following the official training.
### Test
Test (2 GPUs as an example):
```terminal
mim train mmaction configs/repo_actionformer_th14.py --gpus 2 --launcher pytorch --checkpoint work_dirs/repo_actionformer_th14/latest.pth --cfg-options train_dataloader.batch_size=1
```
