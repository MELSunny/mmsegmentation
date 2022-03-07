# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import os
import mmcv
from PIL import Image
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description='generate gif animation')
    parser.add_argument('show_dir', help='inference images folder path')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('group_split', default='_', help='group size of each gif')
    parser.add_argument('out_dir', help='output folder')
    parser.add_argument('--suffix', default='.png', help='suffx of the image file')
    parser.add_argument('--ignore_img', default=0,type=int, help='ignore number of beginning and ending images per group')
    parser.add_argument('--skip_img', default=1,type=int, help='skip images')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    return args


def save_gif(np_imglist,filename,duration=200,loop=0):
    img, *imgs = [Image.fromarray(np_img) for np_img  in np_imglist]
    img.save(fp=filename, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=loop)

def main():
    args = parse_args()
    filelists= [item for item in os.listdir(args.show_dir) if item.endswith(args.suffix)]
    groups=list(set([group.split(args.group_split)[0] for group in filelists]))
    cfg = mmcv.Config.fromfile(args.config)

    for group in groups:
        out_imgs=[]
        ids=[item.split(args.group_split)[1].replace(args.suffix,'')  for item in filelists if item.startswith(group)]
        ids.sort(key=int)
        if args.skip_img!=0:
            ids=ids[args.ignore_img:-args.ignore_img]
        for id in ids[::args.skip_img]:
            ann= mmcv.imread(osp.join(cfg.data.test.data_root, cfg.data.test.ann_dir, group + args.group_split + id + args.suffix))
            img = mmcv.imread(osp.join(cfg.data.test.data_root,cfg.data.test.img_dir,group+args.group_split+id+args.suffix))
            pre_img = mmcv.imread(osp.join(args.show_dir, group + args.group_split+id + args.suffix))
            binary_obj_mask=ann.max(axis=2)
            ann_img = np.where(~np.stack((binary_obj_mask,binary_obj_mask,binary_obj_mask),axis=2).astype(bool),img,img * (1 - args.opacity) + ann * args.opacity)
            ann_img = ann_img.astype(np.uint8)
            three_imgs=np.concatenate((img, np.ones((384,64,3), dtype=np.uint8)*255 ,ann_img, np.ones((384,64,3), dtype=np.uint8)*255, pre_img), axis=1)
            out_imgs.append(three_imgs)
        save_gif(out_imgs,osp.join(args.out_dir,group+'.gif'))



if __name__ == '__main__':
    main()