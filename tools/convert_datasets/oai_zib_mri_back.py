import argparse
import os.path as osp
import os
import mmcv
from functools import partial
import SimpleITK as sitk
from PIL import Image
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OAI ZIB MRI annotations to mmsegmentation format')
    parser.add_argument('--show-dir', help='OAI ZIB MRI predict path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    show_dir = args.show_dir
    if args.out_dir is None:
        out_dir = osp.join(show_dir, '..' ,osp.basename(show_dir)+'_nifti')
    else:
        out_dir = args.out_dir
    mmcv.mkdir_or_exist(osp.join(out_dir))
    id_list=set([a.split('_')[0] for a in os.listdir(args.show_dir) if a.endswith('.png') ])
    for id in id_list:
        if osp.exists(osp.join(out_dir,id+'.nii.gz')):
            print('Exists, Skip ' + id)
        else:
            slices_ids=[int(a.split('_')[1].split('.png')[0]) for a in os.listdir(args.show_dir) if a.startswith(id)]
            check=True
            for s_id in range(0,max(slices_ids)+1):
                if not s_id in slices_ids:
                    check=False
                    print('Can not find '+id+'_'+str(s_id))
            if check:
                mask3d=None
                for s_id in range(0, max(slices_ids) + 1):
                    mask=Image.open(osp.join(show_dir,id+'_'+str(s_id)+'.png'))
                    mask=np.expand_dims(mask,axis=0)
                    if type(mask3d)==np.ndarray:
                        mask3d=np.concatenate((mask3d, mask))
                    else:
                        mask3d=np.asarray(mask)
                mask3d_img=sitk.GetImageFromArray(mask3d)
                sitk.WriteImage(mask3d_img,osp.join(out_dir,id+'.nii.gz'))
            else:
                print('Skip '+ id)

if __name__ == '__main__':
    main()