_base_ = '../setr/setr_vit-base_mla_512x512_160k_b16_oai_zib_mri.py'
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
