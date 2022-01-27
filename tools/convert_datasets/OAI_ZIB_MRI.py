
import argparse
import os.path as osp
import os
import mmcv
from functools import partial
import SimpleITK as sitk
from PIL import Image
import numpy as np

def convert_mhd(mhd_file, in_dir, out_dir):
    data = sitk.ReadImage(osp.join(in_dir, mhd_file))
    np_data=sitk.GetArrayFromImage(data)
    np_data=np.moveaxis(np_data,-1,0)
    np_data=np_data[::-1,::-1,::]
    data=sitk.GetImageFromArray(np_data)
    _out_dir=osp.join(out_dir,mhd_file.replace('.segmentation_masks.mhd',''))
    mmcv.mkdir_or_exist(_out_dir)
    seg_filename = osp.join(_out_dir, 'mask.nii.gz')
    sitk.WriteImage(data,seg_filename)

def convert_dicom(dicom_file, in_dir, out_dir):
    sitk_reader = sitk.ImageSeriesReader()
    dicom_names = sitk_reader.GetGDCMSeriesFileNames(osp.join(in_dir, dicom_file))
    sitk_reader.SetFileNames(dicom_names)
    data = sitk_reader.Execute()
    _out_dir=osp.join(out_dir,osp.basename(osp.dirname(osp.dirname(dicom_file))))
    mmcv.mkdir_or_exist(_out_dir)
    seg_filename = osp.join(_out_dir, 'image.nii.gz')
    sitk.WriteImage(data,seg_filename)

def convert_nifti(nifti_file, in_dir, out_dir):
    image = sitk.ReadImage(osp.join(in_dir, nifti_file,'image.nii.gz'))
    mask = sitk.ReadImage(osp.join(in_dir, nifti_file,'mask.nii.gz'))
    np_image=sitk.GetArrayFromImage(image)
    np_mask=sitk.GetArrayFromImage(mask)
    im_shape = np_image.shape[1:]
    mask_shape = np_mask.shape[1:]
    assert im_shape==mask_shape and im_shape==(384,384)
    for i in range(0,np_image.shape[0]):
        im = Image.fromarray(np.clip(np_image[i],0,255).astype(np.uint8))

        im.save(osp.join(out_dir,'img_dir',nifti_file+'_'+str(i)+'.png'))
    for i in range(0,np_mask.shape[0]):
        im = Image.fromarray(np_mask[i])
        im.putpalette([0, 0, 0,  # Background - Black
                         255, 0, 0,  # Class 1 - Red
                         0, 255, 0,  # Class 2 - Green
                         0, 0, 255,  # Class 3 - Blue
                         255, 255, 0])  # Class 4 - Yellow

        im.save(osp.join(out_dir,'ann_dir',nifti_file+'_'+str(i)+'.png'))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OAI ZIB MRI annotations to mmsegmentation format')
    parser.add_argument('--image_path', help='OAI ZIB MRI image path')
    parser.add_argument('--aug_path', help='OAI ZIB MRI segmentation_masks path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=12, type=int, help='number of process')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    image_path = args.image_path
    aug_path = args.aug_path
    nproc = args.nproc
    if args.out_dir is None:
        out_dir = osp.join(image_path, '..', '..')
    else:
        out_dir = args.out_dir
    if not os.path.exists(osp.join(out_dir,'nifti')):
        mmcv.mkdir_or_exist(osp.join(out_dir,'nifti'))
        oai_mri_paths_file= osp.join(aug_path, '..', 'doc','oai_mri_paths.txt')
        aug_list_dict={}
        with open(oai_mri_paths_file) as f:
            for line in f:
                [id,path] = line.strip().split(": ")
                aug_list_dict[id]=path
        aug_list=list(mmcv.scandir(aug_path, suffix='.mhd'))
        for item in aug_list:
            if not item.replace('.segmentation_masks.mhd','') in aug_list_dict:
                raise Exception
        mmcv.track_parallel_progress(
            partial(convert_mhd, in_dir=aug_path, out_dir=osp.join(out_dir,'nifti')),
            aug_list,
            nproc=nproc)
        mmcv.track_parallel_progress(
            partial(convert_dicom, in_dir=image_path, out_dir=osp.join(out_dir,'nifti')),
            [aug_list_dict[item.replace('.segmentation_masks.mhd','')] for item in aug_list],
            nproc=nproc)

    else:
        print("Found nifti folder")
        mmcv.mkdir_or_exist(osp.join(out_dir, 'dataset'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'dataset','img_dir'))
        mmcv.mkdir_or_exist(osp.join(out_dir, 'dataset','ann_dir'))
    mmcv.track_parallel_progress(
        partial(convert_nifti, in_dir=osp.join(out_dir,'nifti'), out_dir=osp.join(out_dir,'dataset')),
        os.listdir(osp.join(out_dir,'nifti')),
        nproc=nproc)
    image_file_list=os.listdir(osp.join(out_dir, 'dataset','img_dir'))
    list1 = osp.join(aug_path, '..', 'doc', '2foldCrossValidation-List1.txt')
    list2 = osp.join(aug_path, '..', 'doc', '2foldCrossValidation-List2.txt')

    with open(osp.join(out_dir, 'dataset','2foldCrossValidation-List1.txt'),'w') as fout:
        with open(list1) as f:
            for line in f:
                for item in image_file_list:
                    if item.startswith(line.strip()):
                        fout.write(item.replace('.png','') + "\n")
    with open(osp.join(out_dir, 'dataset','2foldCrossValidation-List2.txt'),'w') as fout:
        with open(list2) as f:
            for line in f:
                for item in image_file_list:
                    if item.startswith(line.strip()):
                        fout.write(item.replace('.png','') + "\n")

if __name__ == '__main__':
    main()