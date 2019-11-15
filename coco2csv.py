import argparse
import cv2
import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

def get_coco_file_path(coco_dir, dataset):
    coco_file = os.path.join(coco_dir,'annotations','instances_' + dataset + '.json')
    return coco_file
    

def coco2csv(coco_file, coco_dir, dataset, csv_file):

    # read COCO json file
    coco = COCO(coco_file)

    # get all images
    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    print('found', len(images),'images')

    # get annotations (annotations exist only for images with objects)
    all_annotations = []
    for image in images:
        ann_ids = coco.getAnnIds(imgIds=image['id'], iscrowd=None)
        annotations = coco.loadAnns(ann_ids)
        if annotations:
            all_annotations.extend(annotations)

    # convert annotations to panda dataframe        
    annotations = pd.DataFrame(data = all_annotations)
    print('found', annotations.shape[0],'annotations')

    # split bbox into coordinate columns & convert from x1,y1,w,h to x1,y1,x2,y2
    annotations[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(annotations['bbox'].values.tolist(), index=annotations.index).astype(int)
    annotations = annotations.assign(x2 = annotations['x1'] + annotations['x2'], y2 = annotations['y1'] + annotations['y2'])

    # add image names
    images = pd.DataFrame(images)
    images.rename(columns={'id':'image_id'}, inplace=True)
    images.set_index('image_id')

    annotations_csv = pd.merge(annotations, images, on=['image_id'], how='right') #annotations.join(images, on='image_id')
    annotations_csv = annotations_csv.replace(np.nan, '', regex=True)

    # select only columns required for the csv format, and fix image path
    colnames = ['file_name', 'x1','y1','x2','y2']
    annotations_csv = annotations_csv[colnames]
    annotations_csv['file_name'] = annotations_csv['file_name'].apply(lambda x : os.path.join(coco_dir,'images', dataset, x))

    # write annotations to file
    annotations_csv.to_csv(path_or_buf=csv_file, index=False, header=False, columns=colnames)
    print('exported', annotations_csv.shape[0],'annotations')

def main():
    parser = argparse.ArgumentParser(description='convert a COCO JSON annotations file to a CSV annotations file.')
    parser.add_argument('--coco_dir', required=True, type=str, help='directory where the COCO dataset is stored')
    parser.add_argument('--dataset',  type=str, default="train2017", help='train set or val set')
    parser.add_argument('--csv_file', required=True, type=str, help='output CSV file')

    args = parser.parse_args()
    coco_file = get_coco_file_path(args.coco_dir, args.dataset)
    coco2csv(coco_file, args.coco_dir, args.dataset, args.csv_file)


if __name__== "__main__":
    main()