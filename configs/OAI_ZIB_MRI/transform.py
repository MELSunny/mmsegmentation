from mmcv import Config
import mmseg.datasets.pipelines.compose as compose
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
cfg = Config.fromfile('configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py')
cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.img_scale = (512, 512)
cfg.crop_size = (512, 512)

transforms=[dict(type='Resize', img_scale=cfg.img_scale, ratio_range=(0.5, 2.0))]
test_img='/media/yanwe/1414469C7C0E7D26/Data/OAI-ZIB-MRI/dataset_small/img_dir/9001104_115.png'
test_msk='/media/yanwe/1414469C7C0E7D26/Data/OAI-ZIB-MRI/dataset_small/ann_dir/9001104_115.png'

in_img=dict(img=np.asarray(Image.open(test_img)),gt_semantic_seg=np.asarray(Image.open(test_msk)))
comp=compose.Compose(transforms)
out=comp(in_img)
imgplot = plt.imshow(out['img'])
plt.show()