# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class OAIZIBMRIDataset(CustomDataset):
    """OAI-ZIB-MRI dataset.

    Args:
        split (str): Split txt file for OAI-ZIB-MRI.
    """

    CLASSES = ('background', 'femoral bone', 'femoral cartilage', 'tibial bone', 'tibial cartilage')
    PALETTE = [[0, 0, 0],
               [255, 0, 0],  # Class 1 - Red
               [0, 255, 0],  # Class 2 - Green
               [0, 0, 255],  # Class 3 - Blue
               [255, 255, 0]]  # Class 4 - Yellow

    def __init__(self, split, **kwargs):
        super(OAIZIBMRIDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
