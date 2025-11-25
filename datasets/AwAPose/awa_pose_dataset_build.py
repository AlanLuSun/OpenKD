import os
import torch
import json
import argparse
from PIL import Image
import cv2
import numpy as np
import random
import math
from collections import OrderedDict
import matplotlib.pyplot as plt
from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance
# import sys
# sys.path.append('..')
import pickle


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


def AwA_json_build(anno_root, save_root=None):
    '''
    All the data entries are contained in a dict, which has two keys,
    {
    'classid2name': ['animal1', 'animal2', ..., 'animal35'],  # a list containing class names
    'anns': [{'filename': 'xxx', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
    }  # list
    Each entry in 'anns' is a dict, where its format is as above.
    The path of each image is dataset_root/images/class_name/filename, where the class_name could be retrieved via dataset['classid2name'][classid].
    '''
    dataset_dict = {}

    coco_dataset = pickle.load(open(os.path.join(anno_root, 'dataset.pickle'), 'rb'))
    print('annotation loaded!')

    keypoint_types = coco_dataset['categories'][0]['keypoints']  # 39 keypoint types
    # check keypoint types
    KEYPOINT_TYPES = get_keypoints()
    for i, kp_type in enumerate(keypoint_types):
        if kp_type != KEYPOINT_TYPES[i]:
            print('{} not match {}'.format(kp_type, KEYPOINT_TYPES[i]))
            exit(0)
    print('keypoint types successfully checked!')

    toImgs = {}
    for img_entry in coco_dataset['images']:
        img_id = img_entry['id']
        toImgs[img_id] = img_entry

    category_list = []
    for each_ann in coco_dataset['annotations']:
        category = each_ann['animal']
        if category not in category_list:
            category_list.append(category)
    dataset_dict['classid2name'] = category_list

    anns = []
    for each_ann in coco_dataset['annotations']:
        image_entry = toImgs[each_ann['image_id']]
        im_w, im_h = image_entry['width'], image_entry['height']

        each_entry = {}
        category = each_ann['animal']
        keypoints = each_ann['keypoints']
        each_entry['filename'] = image_entry['filename']
        each_entry['classid'] = category_list.index(category)

        xs = keypoints[0::3]
        ys = keypoints[1::3]
        vs = keypoints[2::3]
        each_entry['part'] = [[xs[i], ys[i], 1] if v != 0 else [0, 0, 0] for i, v in enumerate(vs)]

        x, y, xmax, ymax = each_ann['bbox']  # 'bbox' is equal to 'clean_bbox' but in different forms
        x2, y2, w2, h2 = each_ann['clean_bbox']
        if (x2+w2) != xmax or (y2+h2) != ymax:
            print(False)

        # (xmin, xmax, ymin, ymax)
        each_entry['bbx'] = [max(0, x), min(xmax, im_w-1), max(0, y), min(ymax, im_h-1)]
        # each_entry['bbx'] = [max(0, x2), min(x2+w2, im_w-1), max(0, y2), min(y2+h2, im_h-1)]

        each_entry['w_h'] = [im_w, im_h]
        anns.append(each_entry)

    dataset_dict['anns'] = anns

    if save_root is not None:
        with open(os.path.join(save_root, 'AwAPose_annotations.json'), 'w') as fout:
            json.dump(dataset_dict, fout)
            fout.close()

    return dataset_dict



def AwA_dataset_split(dataset_root, save_root=None):
    '''
    All the data entries are contained in a dict, which has two keys,
    {
    'classid2name': ['animal1', 'animal2', ..., 'animal35'],  # a list containing class names
    'anns': [{'filename': 'xxx', 'classid': int, 'part': [[x, y, is_visible],...], 'bbx': [x_min, x_max, y_min, y_max], 'w_h': [w, h]}, ...]
    }  # list
    Each entry in 'anns' is a dict, where its format is as above.
    The path of each image is dataset_root/images/class_name/filename, where the class_name could be retrieved via dataset['classid2name'][classid].
    '''
    with open(os.path.join(dataset_root, 'AwAPose_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    # build a class_id to image_ids list
    cat2img = {}
    for i in range(len(dataset_dict['anns'])):
        entry = dataset_dict['anns'][i]
        classid = entry['classid']
        if classid not in cat2img:
            cat2img[classid] = []
        cat2img[classid].append(i)

    cat2name = dataset_dict['classid2name']

    support = []
    # val_ref = []
    # val_query = []
    # test_ref = []
    # test_query = []
    # val = []
    test = []

    #---------------------
    train2 = []  # for each species category, split 70% for training and 30% for testing
    test2 = []
    split_ratio = 0.7
    # ---------------------

    # support_cat = []
    # val_cat = []
    # test_cat = []
    for i in range(0, 35):
        img_list = cat2img[i]
        img_num = len(img_list)
        name = cat2name[i]

        if i%7 <= 4:  # partition 5/7 for training categories
            # support_cat.append(name)
            support.extend(img_list)
        # elif i%4 == 0:
        #     # val_cat.append(name)
        #     # val_ref.extend(img_list[:img_num//5])
        #     # val_query.extend(img_list[img_num//5:])
        #     val.extend(img_list)
        elif i%7 == 5 or i%7 ==6:
            # test_cat.append(name)
            # test_ref.extend(img_list[:img_num//5])
            # test_query.extend(img_list[img_num//5:])
            test.extend(img_list)

        # ---------------------
        # for each species category, split 70% for training and 30% for testing
        img_num_sampled = (int)(split_ratio * img_num + 0.5)
        train2_each_species = random.sample(img_list, img_num_sampled)
        test2_each_species = list(set(img_list).difference(set(train2_each_species)))
        train2.extend(train2_each_species)
        test2.extend(test2_each_species)
        # ---------------------

    # copy the instances from dataset_dict to each subsets
    img_id_lists = [support, test, train2, test2]
    # cat_id_lists = [support_cat, val_cat, test_cat]
    subset_lists = []
    for i in range(len(img_id_lists)):
        each_dataset = {}
        # each_dataset['classid2name'] = cat_id_lists[i]
        each_dataset['classid2name'] = dataset_dict['classid2name']
        each_dataset['anns'] = [dataset_dict['anns'][j] for j in img_id_lists[i]]  # copy instances
        # for j in range(len(each_dataset['anns'])):  # update the relative classid in each subsets
        #     classid = each_dataset['anns'][j]['classid']
        #     name = cat2name[classid]
        #     new_classid = each_dataset['classid2name'].index(name)
        #     each_dataset['anns'][j]['classid'] = new_classid
        subset_lists.append(each_dataset)
    subset_support, subset_test, subset_train2, subset_test2 = subset_lists
    if save_root is not None:
        with open(os.path.join(save_root, 'AwAPose_split_train.json'), 'w') as fout:
            json.dump(subset_support, fout)
            fout.close()
        # with open(os.path.join(save_root, 'AwAPose_split_val.json'), 'w') as fout:
        #     json.dump(subset_val, fout)
        #     fout.close()
        with open(os.path.join(save_root, 'AwAPose_split_test.json'), 'w') as fout:
            json.dump(subset_test, fout)
            fout.close()

        # for each species category, split 70% for training and 30% for testing
        with open(os.path.join(save_root, 'AwAPose%.2f.json'%split_ratio), 'w') as fout:
            json.dump(subset_train2, fout)
            fout.close()
        with open(os.path.join(save_root, 'AwAPose%.2f.json'%(1-split_ratio)), 'w') as fout:
            json.dump(subset_test2, fout)
            fout.close()





if __name__ == '__main__':
    # path = '/home/changsheng/LabDatasets/AwA-Pose/Animals_with_Attributes2/JPEGImages'
    # for each_dir in os.listdir(path):
    #     for f in os.listdir(os.path.join(path, each_dir)):
    #         _, ext = os.path.splitext(f)
    #         if ext != '.jpg':
    #             print(ext)
    #             exit(0)
    # print('all jpg')
    # exit(0)

    anno_pickle_root = '/home/changsheng/LabDatasets/AwA-Pose'
    data_root = '/home/changsheng/LabDatasets/AwA-Pose/Animals_with_Attributes2'
    # AwA_json_build(anno_root=anno_pickle_root, save_root=data_root)
    AwA_dataset_split(data_root, save_root=data_root)
    exit(0)


    with open(os.path.join(data_root, 'AwAPose_annotations.json'), 'r') as fin:
        dataset_dict = json.load(fin)
        fin.close()

    classid2name = dataset_dict['classid2name']
    entry = dataset_dict['anns'][0]
    path = os.path.join(data_root, 'JPEGImages', classid2name[entry['classid']], entry['filename'])
    # path = entry['filename']
    print(path)
    keypoints_dict = {kp_type: entry['part'][i] for i, kp_type in enumerate(get_keypoints())}
    # one_kp_type = 'breast'
    # keypoints = {one_kp_type: entry['part'][get_keypoints().index(one_kp_type)]}
    KEYPOINT_TYPES = get_keypoints()
    draw_instance(path, keypoints_dict, KEYPOINT_TYPES, limbs=[], visible_bounds=[entry['bbx'][3] - entry['bbx'][2], entry['bbx'][1] - entry['bbx'][0],  entry['bbx'][0],  entry['bbx'][2]], hightlight_keypoint_types=None, save_root='.', is_show=True)
    print('ok')