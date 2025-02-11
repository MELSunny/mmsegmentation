_base_ = ['./segformer_mit-b0_512x512_160k_oai_zib_mri.py']

# model settings
model = dict(
    pretrained='pretrain/mit_b5.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
