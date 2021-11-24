_base_ = '../swin/upernet_swin_base_patch4_window12_512x512_160k_oai_zib_mri_pretrain_384x384_22K.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
