# Temporal action detection using mmdetection and mmaction2

This repository contains **unoffical** codes of several temporal action detection (TAD) methods that implemented in [open-mmlab](https://github.com/open-mmlab) style. 
[mmengine](https://github.com/open-mmlab/mmengine), [mmcv](https://github.com/open-mmlab/mmcv), [mmdetection](https://github.com/open-mmlab/mmdetection), and [mmaction2](https://github.com/open-mmlab/mmaction2) are the main backends.

> I am **NOT** an employee of open-mmlab, **neither** the author of many of the implemented TAD methods here

# Supported TAD methods
- APN (official)
- DITA (official)
- ActionFormer
- TadTR
- BasicTAD


# Current status (2 Jan 2024)
The repository is still under construction and the readme.md need to be updated.


# Prepare the environment
## Create a mmtad environment
```terminal
conda create -n mmengine python=3.8 -y
conda activate mmengine
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install openmim
mim install mmengine mmdet mmaction2
pip install fvcore future tensorboard pytorchvideo timm
```
You need pay attention to the version compatibility of PyTorch, CUDA and NVIDIA driver 
[link1](https://www.nvidia.com/download/index.aspx), [link2](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html), [link3](https://pytorch.org/get-started/previous-versions/). 


You need pay attention to the installation message of `mmcv` and check if it is something like:
```terminal
Collecting mmcv
Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/mmcv-2.0.0-cp38-cp38-manylinux1_x86_64.whl
```
or 
```terminal
Collecting mmcv
Downloading mmcv-2.1.0.tar.gz (471 kB)
```
The former indicates that there is a **pre-built** version of `mmcv`  corresponding to the PyTorch and CUDA installed in the conda environment. And in this case, everything should just go fine.

While if it's the second case, i.e., it installed `mmcv` using a `.tar.gz` file. It means that there was NO proper pre-build `mmcv` and it was **building the mmcv from the source**. 
In this case, some tricky errors may appear. For example, it could raise a `CUDA version mismatch` error if your versions of the system-wide CUDA and the conda-wide CUDA are mismatched. 
You could check the available pre-built `mmcv` in [this page](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip).


## Add current path into the Python path
Add the root directory to the Python path, otherwise you need add `PYTHONPATH=$PWD:$PYTHONPATH` before every command:
```terminal
cd DITA
export PYTHONPATH=$PWD:$PYTHONPATH
```
Note that once you close the terminal, you need re-run the above command as it is a temporary setting.

## A recipe of commands
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
# Reproduce BasicTAD(TAD)
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
