from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config
import os.path as osp
from mmseg.apis import set_random_seed
data_root = '/media/yanwe/1414469C7C0E7D26/Data/OAI-ZIB-MRI/dataset_full'
img_dir = 'img_dir'
ann_dir = 'ann_dir'
@DATASETS.register_module()
class StandfordBackgroundDataset(CustomDataset):
  CLASSES =( 'femoral bone','femoral cartilage','tibial bone','tibial cartilage')
  PALETTE = [[255, 0, 0],  # Class 1 - Red
             [0, 255, 0],  # Class 2 - Green
             [0, 0, 255],  # Class 3 - Blue
             [255, 255, 0]]  # Class 4 - Yellow
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None

cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r101-d8_512x512_160k_ade20k.py')
# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.num_classes = 4
cfg.model.auxiliary_head.num_classes = 4
cfg.dataset_type = 'StandfordBackgroundDataset'
cfg.data_root = data_root
cfg.data.samples_per_gpu = 3
cfg.data.workers_per_gpu= 6
cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (512, 512)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cfg.val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
cfg.test_pipeline=None
cfg.data.test=None
cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = osp.join(data_root,'2foldCrossValidation-List1.txt')

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.val_pipeline
cfg.data.val.split = osp.join(data_root,'2foldCrossValidation-List2.txt')


#cfg.load_from = 'checkpoints/deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth'
cfg.work_dir = osp.join(data_root,'deeplabv3plus')
cfg.log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False),dict(type='TensorboardLoggerHook',by_epoch=False,log_dir=cfg.work_dir) ])
cfg.runner.max_iters =  int(40480/cfg.data.samples_per_gpu*50)
cfg.evaluation.interval = 500
cfg.checkpoint_config.interval = 500
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import mmcv
# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())