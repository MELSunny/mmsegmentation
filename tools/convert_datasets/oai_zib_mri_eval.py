import argparse
import os.path as osp
import os

import SimpleITK as sitk

import numpy as np
import json
import pandas as pd
CLASSES = ('background', 'femoral bone', 'femoral cartilage', 'tibial bone', 'tibial cartilage')
pd_classes=('femoral bone',  'tibial bone' ,'femoral cartilage', 'tibial cartilage')
pd_metrics=('dice','asd','rsd','msd','vd','voe')
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OAI ZIB MRI annotations to mmsegmentation format')
    parser.add_argument('--nifti-path', help='OAI ZIB MRI nifti path')
    parser.add_argument('-o', '--out_dir', help='output from oai_zib_mri_back path')
    args = parser.parse_args()
    return args

def statisticize(result_dict):
    result_mean = pd.DataFrame(columns=pd_metrics,index=pd_classes)
    result_std = pd.DataFrame(columns=pd_metrics,index=pd_classes)
    for cla in pd_classes:
        for metric in pd_metrics:
            result_list=[]
            for id in result_dict:
                result_list.append(result_dict[id][cla][metric])
            result_mean[metric][cla]=np.mean(result_list)
            result_std[metric][cla] = np.std(result_list)

    print(result_mean)
    print(result_std)


def main():
    args = parse_args()
    if osp.exists(args.out_dir+'.json'):
        with open(args.out_dir+'.json', 'r') as f:
            data = json.load(f)
            statisticize(data)
            return
    pred_segs =[file for file in os.listdir(args.out_dir) if file.endswith('nii.gz')]
    quality = dict()
    for pred_seg in pred_segs:
        id=pred_seg.split('.')[0]
        print(id)
        gt_seg_path=osp.join(args.nifti_path,id,'mask.nii.gz')
        image_path=osp.join(args.nifti_path,id,'image.nii.gz')
        pred_seg_path=osp.join(args.out_dir,pred_seg)
        labelTrue=sitk.ReadImage(gt_seg_path)
        image=sitk.ReadImage(image_path)
        labelPred=sitk.ReadImage(pred_seg_path)
        labelPred.SetSpacing(image.GetSpacing())
        labelTrue.SetSpacing(image.GetSpacing())
        result_classes=dict()
        for i in range(0,len(CLASSES)):
            result_class=dict()
            dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
            dicecomputer.Execute(labelTrue ==i, labelPred ==i)
            result_class["dice"]=dicecomputer.GetDiceCoefficient()
            result_class["voe"]=1-dicecomputer.GetDiceCoefficient()/(2-dicecomputer.GetDiceCoefficient())
            pred_statcomputer=sitk.StatisticsImageFilter()
            pred_statcomputer.Execute(labelPred ==i)
            true_statcomputer = sitk.StatisticsImageFilter()
            true_statcomputer.Execute(labelTrue ==i)
            result_class["vd"]=(pred_statcomputer.GetSum() -true_statcomputer.GetSum())/true_statcomputer.GetSum()


            reference_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(labelTrue ==i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(labelTrue ==i)

            statistics_image_filter = sitk.StatisticsImageFilter()

            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())


            segmented_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(labelPred ==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(labelPred ==i)

            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
            all_surface_distances = seg2ref_distances + ref2seg_distances
            result_class["asd"]= np.mean(all_surface_distances)
            result_class["rsd"]= np.sqrt(np.mean(np.array(all_surface_distances)**2))
            result_class["msd"]= np.max(all_surface_distances)
            result_classes[CLASSES[i]]=result_class
        quality[id]=result_classes

    result = json.dumps(quality)
    statisticize(quality)

    f = open(args.out_dir+".json","w")

    # write json object to file
    f.write(result)

    # close file
    f.close()

if __name__ == '__main__':
    main()