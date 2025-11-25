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

# 11 keypoints for NABird dataset
def get_keypoints():
    keypoints = [
        'bill',  # namely beak
        'crown',
        'nape',
        'left eye',
        'right eye',
        'belly',
        'breast',
        'back',
        'tail',
        'left wing',
        'right wing',
    ]
    return keypoints

def convert_to_coco():
    '''
        For 'nabird_annotations.json', all the data entries are contained in a dict, which has four keys,
            {
            'classid2name': {classid0: 'bird0', classid1: 'bird1', ...},  # a dict containing pairs of class_id: class_name, all class_id is child class, and this dict is subset of fullclassid2name
            'fullclassid2name': {classid0: 'bird0', classid1: 'bird1', ...}, # a dict containing both child classes and parent classes
            'hierarchy': {child_class_id: parent_class_id},  # used to retrieve each class's parent
            'anns': [{'filename': 'classfolder/filename.jpg', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
            }  # list
        Each entry in 'anns' is a dict, where its format is as above.
        The path of each image is dataset_root/images/classid_folder/filename.

        We convert above format to COCO format.
    '''
    root = '/home/changsheng/LabDatasets/BirdDataset/NABird/nabirds'
    save_root = '/home/changsheng/LabDatasets/BirdDataset/NABird/annotations-coco'

    with open(os.path.join(root, 'nabird_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    dataset_coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],  # this list contains child class that is subset of fullclassid2name
        "fullclassid2name": {},  # additionally added for NABird. a dict containing both child classes and parent classes, e.g., {classid0: 'bird0', classid1: 'bird1',...}
        "hierarchy": {}, # additionally added for NABird. used to retrieve each class's parent, e.g., {child_class_id: parent_class_id, ...}
    }

    dataset_coco['info']['description'] = 'NABird dataset (555 birds)'
    species = dataset_dict['classid2name']
    for id, each_species in species.items():
        dataset_coco['categories'].append({
            'id': int(id),  # 295-based for NABird's category id
            'name': each_species,
            'supercategory': "birds",
            'keypoints': get_keypoints(),  # a list of ordered keypoint names (important)
            'skeleton': [],  # a list of limbs, namely keypoint id pairs (id is 1-based)(not important)
        })

    fullclasses = dataset_dict['fullclassid2name']  # contain both parent classes and child classes
    for id, each_class_name in fullclasses.items():
        dataset_coco['fullclassid2name'][int(id)] = each_class_name
    hierarchy = dataset_dict['hierarchy']
    for child_id, parent_id in hierarchy.items():
        dataset_coco['hierarchy'][int(child_id)] = parent_id

    # cat2id = {}
    # for cat_entry in dataset_coco['categories']:
    #     cat_name = cat_entry['name']
    #     cat_id = cat_entry['id']
    #     cat2id[cat_name] = cat_id

    im_id = 1  # 1-based, coco required
    anno_id = 1  # 1-based, coco required
    im2id = {}
    for one_sample in dataset_dict['anns']:
        filename = one_sample['filename']
        cat_id = one_sample['classid']  # e.g., 295
        keypoints = one_sample['part']  # [[x, y, isvisible], ...]
        bbx = one_sample['bbx']  # [x_min, x_max, y_min, y_max]
        w_h = one_sample['w_h']  # [int, int], image width and height

        if (filename in im2id.keys()) == False:
            im2id[filename] = im_id
            # add a new image entry
            im_entry = {
                'id': im_id,
                'file_name': filename,  # e.g., '0817/0000139e21dc4d0cbfe14cae3c85c829.jpg'
                'width': w_h[0],
                'height': w_h[1],
                'im_root': 'nabirds/images'  # this item is added by me
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
            'image_id': im2id[filename],
            'category_id': cat_id,
            'bbox': [xmin, ymin, bbx_w, bbx_h],  # [xmin, ymin, width, height]
            'area': bbx_w * bbx_h,
            'keypoints': keypoints_flatten,      # [x, y, isvisible, ...], note it is 1D list
            'num_keypoints': num_keypoints,
            'iscrowd': 0,
        }
        dataset_coco['annotations'].append(anno_entry)
        anno_id += 1

    with open(os.path.join(save_root, 'nabird_annotations.json'), 'w') as fout:
        json.dump(dataset_coco, fout)
        fout.close()


def nab_dataset_split():
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
            "fullclassid2name": {},  # additionally added for NABird. a dict containing both child classes and parent classes, e.g., {classid0: 'bird0', classid1: 'bird1',...}
            "hierarchy": {}, # additionally added for NABird. used to retrieve each class's parent, e.g., {child_class_id: parent_class_id, ...}
        }
        dataset_new['info'] = cocoGt.dataset['info']
        dataset_new['licenses'] = cocoGt.dataset['licenses']
        for id in cat_ids:
            dataset_new['categories'].append(cocoGt.cats[id])
        for id in img_ids:
            dataset_new['images'].append(cocoGt.imgs[id])
        for id in anno_ids:
            dataset_new['annotations'].append(cocoGt.anns[id])
        dataset_new['fullclassid2name'] = cocoGt.dataset['fullclassid2name']  # additionally added for NABird.
        dataset_new['hierarchy'] = cocoGt.dataset['hierarchy']  # additionally added for NABird.
        return dataset_new

    coco_anno_root = '/home/changsheng/LabDatasets/BirdDataset/NABird/annotations-coco/'
    coco_anno_path = os.path.join(coco_anno_root, 'nabird_annotations.json')
    cocoGt = COCO(coco_anno_path)
    cat_ids = cocoGt.getCatIds()
    cat_ids.sort()

    train = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    val   = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    test  = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}

    split_ratio = 0.7  # supervised setting
    train2= {'cat_ids': [], 'img_ids': [], 'anno_ids': []}
    test2 = {'cat_ids': [], 'img_ids': [], 'anno_ids': []}

    for i, id in enumerate(cat_ids):
        img_ids = cocoGt.getImgIds(catIds=id)
        anno_ids= cocoGt.getAnnIds(catIds=id)
        cat_entry = cocoGt.loadCats(ids=id)
        cat_name = cat_entry[0]['name']

        if (i%5 == 0) or (i%5 == 1) or (i%5 == 2):  # in order to have same partition with PoseNorm-Fewshot paper
            train['cat_ids'].append(id)
            train['img_ids'].extend(img_ids)
            train['anno_ids'].extend(anno_ids)
        elif (i % 5 == 3):
            val['cat_ids'].append(id)
            val['img_ids'].extend(img_ids)
            val['anno_ids'].extend(anno_ids)
        elif (i % 5 == 4):
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
            'nabird_split_train.json',
            'nabird_split_val.json',
            'nabird_split_test.json',
            'nabird0.70.json',
            'nabird0.30.json',
        ]
    for i, each_split in enumerate([train, val, test, train2, test2]):
        each_subset = copy_given_ids(cocoGt, cat_ids=each_split['cat_ids'], img_ids=each_split['img_ids'],
                                     anno_ids=each_split['anno_ids'])

        save_path = os.path.join(coco_anno_root, json_files[i])
        with open(save_path, 'w') as fout:
            json.dump(each_subset, fout)
            fout.close()

    print('split finished!')


if __name__=='__main__':
    # convert_to_coco()
    # nab_dataset_split()
    # exit(0)

    origin_anno_root = '/home/changsheng/LabDatasets/BirdDataset/NABird/nabirds/'
    origin_dataset = json.load(open(os.path.join(origin_anno_root, 'nabird_annotations.json'), 'r'))


    im_root = '/home/changsheng/LabDatasets/BirdDataset/NABird/nabirds/images'
    coco_anno_root = '/home/changsheng/LabDatasets/BirdDataset/NABird/annotations-coco'
    dataset_coco = COCO(os.path.join(coco_anno_root, 'nabird_annotations.json'))
    # dataset_coco = COCO(os.path.join(coco_anno_root, 'nabird_split_test.json'))
    print(dataset_coco.dataset['categories'][:3])

    # {'id': 295, 'name': 'Common Eider (Adult male)', 'supercategory': 'birds',
    #  'keypoints': ['bill', 'crown', 'nape', 'left eye', 'right eye', 'belly', 'breast', 'back', 'tail', 'left wing',
    #                'right wing'], 'skeleton': []}

    cat_id = dataset_coco.getCatIds(catNms='Common Eider (Adult male)')
    img_ids = dataset_coco.getImgIds(catIds=cat_id)
    ann_ids = dataset_coco.getAnnIds(catIds=cat_id)
    anns = dataset_coco.loadAnns(ann_ids)
    NN = 2
    im_entry = dataset_coco.loadImgs([anns[NN]['image_id']])
    I = Image.open(os.path.join(im_root, im_entry[0]['file_name'])).convert('RGB')
    plt.imshow(I)
    plt.axis('off')
    dataset_coco.showAnns([anns[NN]], draw_bbox=True)
    plt.show()

