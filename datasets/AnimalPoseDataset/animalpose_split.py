import copy
import os
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import math
import json
from PIL import Image
import random

from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance
from datasets.coco import COCO


# 20 keypoints for animal pose dataset
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'left eye',
        'right eye',
        'left ear',
        'right ear',
        'nose',
        'throat',
        'withers',
        'tail',
        'left-front leg',
        'right-front leg',
        'left-back leg',
        'right-back leg',
        'left-front knee',
        'right-front knee',
        'left-back knee',
        'right-back knee',
        'left-front paw',
        'right-front paw',
        'left-back paw',
        'right-back paw'
    ]
    return keypoints

# 20 defined limbs
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('nose'), keypoints.index('left eye')],
        [keypoints.index('nose'), keypoints.index('right eye')],
        [keypoints.index('left eye'), keypoints.index('left ear')],
        [keypoints.index('right eye'), keypoints.index('right ear')],
        [keypoints.index('withers'), keypoints.index('tail')],
        [keypoints.index('withers'), keypoints.index('left-front leg')],
        [keypoints.index('left-front leg'), keypoints.index('left-front knee')],
        [keypoints.index('left-front knee'), keypoints.index('left-front paw')],
        [keypoints.index('withers'), keypoints.index('right-front leg')],
        [keypoints.index('right-front leg'), keypoints.index('right-front knee')],
        [keypoints.index('right-front knee'), keypoints.index('right-front paw')],
        [keypoints.index('tail'), keypoints.index('left-back leg')],
        [keypoints.index('left-back leg'), keypoints.index('left-back knee')],
        [keypoints.index('left-back knee'), keypoints.index('left-back paw')],
        [keypoints.index('tail'), keypoints.index('right-back leg')],
        [keypoints.index('right-back leg'), keypoints.index('right-back knee')],
        [keypoints.index('right-back knee'), keypoints.index('right-back paw')],
        # [keypoints.index('withers'), keypoints.index('left ear')],
        # [keypoints.index('withers'), keypoints.index('right ear')],
        # [keypoints.index('withers'), keypoints.index('nose')],
        [keypoints.index('throat'), keypoints.index('withers')],
        # [keypoints.index('throat'), keypoints.index('left eye')],
        # [keypoints.index('throat'), keypoints.index('right eye')]
        [keypoints.index('throat'), keypoints.index('nose')]
    ]
    return kp_lines


def supervised_task_split(anno_path, anno_save_root, split_ratio=0.7):
    # split a COCO-format annotation file into train/test file according to a split_ratio
    # the split occurs in annotation entries per object category
    def copy_given_ids(cocoGt, cat_ids, img_ids, anno_ids):
        dataset_new = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        dataset_new['info'] = cocoGt.dataset['info']
        dataset_new['licenses'] = cocoGt.dataset['licenses']
        for id in cat_ids:
            dataset_new['categories'].append(cocoGt.cats[id])
        for id in img_ids:
            dataset_new['images'].append(cocoGt.imgs[id])
        for id in anno_ids:
            dataset_new['annotations'].append(cocoGt.anns[id])
        return dataset_new

    cocoGt = COCO(anno_path)

    _, anno_name = os.path.split(anno_path)
    anno_name_without_ext, ext = os.path.splitext(anno_name)
    train_annotation_path = os.path.join(anno_save_root, anno_name_without_ext + '%.2f.json' % split_ratio)
    test_annotation_path = os.path.join(anno_save_root, anno_name_without_ext + '%.2f.json' % (1 - split_ratio))

    # split annotation entries per category by a ratio, e.g., 0.7:0.3
    # note here we can divide dataset by annotations or images. We divide by annotations.
    cat_ids = cocoGt.getCatIds()
    im_ids = cocoGt.getImgIds()
    train_anno_ids, test_anno_ids = [], []
    for id in cat_ids:
        anno_ids = cocoGt.getAnnIds(catIds=id)
        index_set = set(anno_ids)
        num_train = round(split_ratio * len(anno_ids))
        train_ids_per_cat = random.sample(index_set, num_train)
        test_ids_per_cat = index_set.difference(train_ids_per_cat)
        train_anno_ids.extend(train_ids_per_cat)
        test_anno_ids.extend(test_ids_per_cat)

    dataset_train_split = copy_given_ids(cocoGt, cat_ids, im_ids, train_anno_ids)
    dataset_test_split = copy_given_ids(cocoGt, cat_ids, im_ids, test_anno_ids)
    json.dump(dataset_train_split, open(train_annotation_path, 'w'))
    json.dump(dataset_test_split, open(test_annotation_path, 'w'))


if __name__=='__main__1':
    anno_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt_coco'
    im_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'

    anno_path = os.path.join(anno_root, 'animal_pose_dataset.json')
    supervised_task_split(anno_path, anno_save_root=anno_root, split_ratio=0.7)


if __name__=='__main__':  # use COCO API to show an examlle
    anno_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt_coco'
    im_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'

    # with open(os.path.join(anno_root, 'animal_pose_dataset0.7.json'), 'r') as fin:
    #     dataset = json.load(fin)
    #     fin.close()

    coco_train = COCO(os.path.join(anno_root, 'animal_pose_dataset0.70.json'))
    coco_test  = COCO(os.path.join(anno_root, 'animal_pose_dataset0.30.json'))

    cat_id = coco_train.getCatIds(catNms=['cat'])
    img_ids = coco_train.getImgIds(catIds=cat_id)
    ann_ids = coco_train.getAnnIds(catIds=cat_id)
    anns = coco_train.loadAnns(ann_ids)
    NN = 3
    im_entry = coco_train.loadImgs([anns[NN]['image_id']])
    I = Image.open(os.path.join(im_root, im_entry[0]['file_name'])).convert('RGB')
    plt.imshow(I)
    plt.axis('off')
    coco_train.showAnns([anns[NN]], draw_bbox=True)
    plt.show()

    cat_id2  = coco_test.getCatIds(catNms=['cat'])
    img_ids2 = coco_test.getImgIds(catIds=cat_id2)
    ann_ids2 = coco_test.getAnnIds(catIds=cat_id2)
    anns2 = coco_test.loadAnns(ann_ids2)

