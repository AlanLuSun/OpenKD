import os
import random

import torch
import json
import argparse
from PIL import Image
import cv2
import numpy as np
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance
from datasets.coco import COCO
# import sys
# sys.path.append('..')

# 39 keypoints in AwA Pose. Some animal does not have all keypoints.
def get_keypoints():
    keypoints = [
        # '_background_',
        'nose',
        'upper_jaw',
        'lower_jaw',
        'mouth_end_right',
        'mouth_end_left',
        'right_eye',
        'right_earbase',
        'right_earend',
        'right_antler_base',
        'right_antler_end',
        'left_eye',
        'left_earbase',
        'left_earend',
        'left_antler_base',
        'left_antler_end',
        'neck_base',
        'neck_end',
        'throat_base',
        'throat_end',
        'back_base',
        'back_end',
        'back_middle',
        'tail_base',
        'tail_end',
        'front_left_thai',
        'front_left_knee',
        'front_left_paw',
        'front_right_thai',
        'front_right_paw',
        'front_right_knee',
        'back_left_knee',
        'back_left_paw',
        'back_left_thai',
        'back_right_thai',
        'back_right_paw',
        'back_right_knee',
        'belly_bottom',
        'body_middle_right',
        'body_middle_left',
    ]
    return keypoints


# limbs defined in AwA pose dataset
skeleton = [(10, 5), (10, 11), (5, 6), (6, 7), (11, 12), (0, 5), (0, 10), (1, 2), (3, 4), (6, 15), (11, 15), (15, 16),
            (16, 19), (19, 21), (21, 20), (20, 22), (22, 23), (2, 17), (17, 18), (26, 25), (25, 24), (28, 29), (29, 27),
            (31, 30), (30, 32), (34, 35), (35, 33), (36, 27), (36, 24), (36, 32), (36, 33), (37, 27), (37, 33),
            (38, 32), (38, 24), (38, 19), (38, 20), (37, 19), (37, 20), (18, 24), (18, 27), (6, 8), (8, 9), (11, 13),
            (13, 14)]

# symmetric keypoint's ids
flip_pairs = [[4, 3], [10, 5], [11, 6], [12, 7], [13, 8], [14, 9], [24, 27], [25, 29], [26, 28], [30, 35], [31, 34], [32, 33]]

upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
lower_body_ids = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]


def convert_to_coco():
    '''
        All the data entries are contained in a dict, which has two keys,
        {
        'classid2name': ['animal1', 'animal2', ..., 'animal35'],  # a list containing class names
        'anns': [{'filename': 'xxx', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
        }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/class_name/filename, where the class_name could be retrieved via dataset['classid2name'][classid].

        We convert above format to COCO format.
    '''
    root = '/home/changsheng/LabDatasets/AwA-Pose/Animals_with_Attributes2/'
    save_root = '/home/changsheng/LabDatasets/AwA-Pose/AwAPose-annotations-coco'

    with open(os.path.join(root, 'AwAPose_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    dataset_coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    dataset_coco['info']['description'] = 'AwA pose dataset (35 animals annotated out of 50)'
    species = dataset_dict['classid2name']
    for i, each_species in enumerate(species):
        dataset_coco['categories'].append({
            'id': i + 1,  # 1-based, required by coco format
            'name': each_species,
            'supercategory': "animals",
            'keypoints': get_keypoints(),  # a list of ordered keypoint names (important)
            'skeleton': [],  # a list of limbs, namely keypoint id pairs (id is 1-based)(not important)
        })

    cat2id = {}
    for cat_entry in dataset_coco['categories']:
        cat_name = cat_entry['name']
        cat_id = cat_entry['id']
        cat2id[cat_name] = cat_id

    im_id = 1  # 1-based, coco required
    anno_id = 1  # 1-based, coco required
    im2id = {}
    for one_sample in dataset_dict['anns']:
        filename = one_sample['filename']
        category = species[one_sample['classid']]  # e.g., cat
        keypoints = one_sample['part']  # [[x, y, isvisible], ...]
        bbx = one_sample['bbx']  # [x_min, x_max, y_min, y_max]
        w_h = one_sample['w_h']  # [int, int], image width and height

        file_name_tmp = os.path.join(category, filename)  # e.g., 'cat/cat1.jpg'
        if (file_name_tmp in im2id.keys()) == False:
            im2id[file_name_tmp] = im_id
            # add a new image entry
            im_entry = {
                'id': im_id,
                'file_name': file_name_tmp,  # e.g., 'cat/cat1.jpg'
                'width': w_h[0],
                'height': w_h[1],
                'im_root': 'Animals_with_Attributes2/JPEGImages'  # this item is added by me
            }
            dataset_coco['images'].append(im_entry)
            im_id += 1
        else:
            pass

        # add a new annotation entry
        xmin, xmax, ymin, ymax = bbx
        bbx_w, bbx_h = xmax-xmin+1, ymax-ymin+1
        np_kps = np.array(keypoints)
        num_keypoints = int(np.sum(np_kps[:, 2] > 0))
        keypoints_flatten = [e for sublist in keypoints for e in sublist]
        anno_entry = {
            'id': anno_id,
            'image_id': im2id[file_name_tmp],
            'category_id': cat2id[category],
            'bbox': [xmin, ymin, bbx_w, bbx_h],  # [xmin, ymin, width, height]
            'area': bbx_w * bbx_h,
            'keypoints': keypoints_flatten,      # [x, y, isvisible, ...], note it is 1D list
            'num_keypoints': num_keypoints,
            'iscrowd': 0,
        }
        dataset_coco['annotations'].append(anno_entry)
        anno_id += 1

    with open(os.path.join(save_root, 'AwAPose_annotations.json'), 'w') as fout:
        json.dump(dataset_coco, fout)
        fout.close()


def awa_dataset_split():
    # 1. split COCO-format annotation file into train/val/test DISJOINT species for FSKD
    # 2. split COCO-format annotation file into train/test sets for supervised task, according to split ratio. This split
    # occurs in annotation entries per object category
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

    coco_anno_root = '/home/changsheng/LabDatasets/AwA-Pose/AwAPose-annotations-coco'
    coco_anno_path = os.path.join(coco_anno_root, 'AwAPose_annotations.json')
    cocoGt = COCO(coco_anno_path)
    cat_ids = cocoGt.getCatIds()

    train = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    # val   = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    test  = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}

    split_ratio = 0.7  # supervised setting
    train2= {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    test2 = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}

    for i, id in enumerate(cat_ids):
        img_ids = cocoGt.getImgIds(catIds=id)
        anno_ids= cocoGt.getAnnIds(catIds=id)
        cat_entry = cocoGt.loadCats(ids=id)
        cat_name = cat_entry[0]['name']

        if (i%7 <= 4):  # partition 5/7 for training categories
            train['cat_ids'].append(id)
            train['img_ids'].extend(img_ids)
            train['anno_ids'].extend(anno_ids)
        elif (i % 7 == 5) or (i % 7 == 6):
            test['cat_ids'].append(id)
            test['img_ids'].extend(img_ids)
            test['anno_ids'].extend(anno_ids)

        # ---------------------
        # for each species category, split 70% for training and 30% for testing.
        # note here we can divide dataset by annotations or images. We divide by annotations.
        anno_num = len(anno_ids)
        anno_num_sampled = (int)(split_ratio * anno_num + 0.5)
        train2_each_species = random.sample(anno_ids, anno_num_sampled)
        test2_each_species = list(set(anno_ids).difference(set(train2_each_species)))
        train2['anno_ids'].extend(train2_each_species)
        test2['anno_ids'].extend(test2_each_species)

        train2['cat_ids'].append(id)
        train2['img_ids'].extend(img_ids)
        test2['cat_ids'].append(id)
        test2['img_ids'].extend(img_ids)
        # ---------------------

    json_files = [
            'AwAPose_split_train.json',
            'AwAPose_split_test.json',
            'AwAPose0.70.json',
            'AwAPose0.30.json',
        ]
    for i, each_split in enumerate([train, test, train2, test2]):
        each_subset = copy_given_ids(cocoGt, cat_ids=each_split['cat_ids'], img_ids=each_split['img_ids'],
                                     anno_ids=each_split['anno_ids'])

        save_path = os.path.join(coco_anno_root, json_files[i])
        with open(save_path, 'w') as fout:
            json.dump(each_subset, fout)
            fout.close()

    print('split finished!')


if __name__=='__main__':
    # convert_to_coco()
    # awa_dataset_split()
    # exit(0)

    origin_anno_root = '/home/changsheng/LabDatasets/AwA-Pose/Animals_with_Attributes2/'
    origin_dataset = json.load(open(os.path.join(origin_anno_root, 'AwAPose_annotations.json'), 'r'))
    # origin_dataset = json.load(open(os.path.join(origin_anno_root, 'AwAPose0.70.json'), 'r'))


    im_root = '/home/changsheng/LabDatasets/AwA-Pose/Animals_with_Attributes2/JPEGImages'
    coco_anno_root = '/home/changsheng/LabDatasets/AwA-Pose/AwAPose-annotations-coco'
    dataset_coco = COCO(os.path.join(coco_anno_root, 'AwAPose_annotations.json'))
    # dataset_coco = COCO(os.path.join(coco_anno_root, 'AwAPose_split_test.json'))
    # dataset_coco = COCO(os.path.join(coco_anno_root, 'AwAPose0.70.json'))
    print(dataset_coco.dataset['categories'][:3])

    cat_id = dataset_coco.getCatIds(catNms=['antelope'])  # 'antelope', 'dalmatian'
    img_ids = dataset_coco.getImgIds(catIds=cat_id)
    ann_ids = dataset_coco.getAnnIds(catIds=cat_id)
    anns = dataset_coco.loadAnns(ann_ids)
    NN = 0
    im_entry = dataset_coco.loadImgs([anns[NN]['image_id']])
    I = Image.open(os.path.join(im_root, im_entry[0]['file_name'])).convert('RGB')
    plt.imshow(I)
    plt.axis('off')
    dataset_coco.showAnns([anns[NN]], draw_bbox=True)
    plt.show()