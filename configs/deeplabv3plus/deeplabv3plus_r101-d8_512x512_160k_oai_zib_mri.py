_base_ = './deeplabv3plus_r50-d8_512x512_160k_oai_zib_mri.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
