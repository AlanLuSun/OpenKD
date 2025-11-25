import copy
import os
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import math
import json
from PIL import Image

from datasets.dataset_utils import CocoColors, draw_skeletons, draw_instance


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


if __name__=='__main__2':  # an example to read previous-format json
    # json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/PASCAL2011_animal_annotation_json/' + 'horse.json'
    json_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt' + '/cat.json'
    with open(json_path, 'r') as fin:
        samples = json.load(fin)
        fin.close()

        one_sample = samples[2]  # choose one sample
        filename = one_sample['filename']
        category = one_sample['category']
        keypoints = one_sample['keypoints']
        visible_bounds = one_sample['bbx']
        image_path = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images/' + category + '/' + filename
        KEYPOINT_TYPES = get_keypoints()
        keypoints_dict = OrderedDict(zip(KEYPOINT_TYPES, keypoints))
        LIMBS = kp_connections(KEYPOINT_TYPES)
        draw_instance(image_path, keypoints_dict, KEYPOINT_TYPES, limbs=LIMBS, visible_bounds=visible_bounds,
                      hightlight_keypoint_types=None, is_show=True, save_root='.')
        # if (filename == 'sh1.jpg'):
        #     print('here')
        print(filename)
        print(one_sample)
        # print(filename, category, keypoints, visible_bounds)


if __name__=='__main__3':  # an example to load COCO json file
    p = "/home/changsheng/LabDatasets/MSCOCO/coco2017/annotations/person_keypoints_train2017.json"
    with open(p, 'r') as fin:
        samples = json.load(fin)
        fin.close()
    print(samples['annotations'][0])
    print(samples['images'][0])
    print(samples['categories'][0])

    # example of each category:
    # {'supercategory': 'person', 'id': 1, 'name': 'person',
    #  'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    #                'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    #                'right_knee', 'left_ankle', 'right_ankle'],
    #  'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
    #               [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}


if __name__=='__main__4':  # TODO: convert current json file to coco format
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    dataset['info']['description'] = 'animal pose dataset (5 animals)'

    species = ['cat', 'dog', 'cow', 'horse', 'sheep']
    skeletons = kp_connections(get_keypoints())    # 0-based
    skeletons = (np.array(skeletons) + 1).tolist() # 1-based, required by coco format
    for i, each_species in enumerate(species):
        dataset['categories'].append({
            'id': i+1,  # 1-based, required by coco format
            'name': each_species,
            'supercategory': "four-footed mammal",
            'keypoints': get_keypoints(),  # a list of ordered keypoint names (important)
            'skeleton': skeletons,  # a list of limbs, namely keypoint id pairs (id is 1-based)(not important)
        })

    cat2id = {}
    for cat_entry in dataset['categories']:
        cat_name = cat_entry['name']
        cat_id = cat_entry['id']
        cat2id[cat_name] = cat_id

    root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt'
    im_id = 1    # 1-based, coco required
    anno_id = 1  # 1-based, coco required
    for i, each_species in enumerate(species):
        json_path = os.path.join(root, each_species + '.json')
        with open(json_path, 'r') as fin:
            samples = json.load(fin)
            fin.close()

        im_names = []
        im2id = {}
        for one_sample in samples:
            filename = one_sample['filename']   # e.g., cat1.jpg
            category = one_sample['category']   # e.g., cat
            keypoints = one_sample['keypoints'] # [[x, y, isvisible], ...]
            visible_bounds = one_sample['bbx']  # [height, width, xmin, ymin]
            w_h = one_sample['w_h']             # [int, int], image width and height

            if (filename in im_names) == False:
                im_names.append(filename)
                im2id[filename] = im_id
                # add a new image entry
                im_entry = {
                    'id': im_id,
                    'file_name': os.path.join(category, filename),  # e.g., 'cat/cat1.jpg'
                    'width':  w_h[0],
                    'height': w_h[1],
                    'im_root': 'Animal_Dataset_Combined/images'  # this item is added by me
                }
                dataset['images'].append(im_entry)
                im_id += 1
            else:
                pass

            # add a new annotation entry
            bbx_h, bbx_w, xmin, ymin = visible_bounds
            np_kps = np.array(keypoints)
            num_keypoints = int(np.sum(np_kps[:, 2] > 0))
            keypoints_flatten = [e for sublist in keypoints for e in sublist]
            anno_entry = {
                'id': anno_id,
                'image_id': im2id[filename],
                'category_id': cat2id[category],
                'bbox': [xmin, ymin, bbx_w, bbx_h],  # [xmin, ymin, width, height]
                'area': bbx_w * bbx_h,
                'keypoints': keypoints_flatten,      # [x, y, isvisible, ...], note it is 1D list
                'num_keypoints': num_keypoints,
                'iscrowd': 0,
            }
            dataset['annotations'].append(anno_entry)
            anno_id += 1

    save_p = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt_coco/animal_pose_dataset.json'
    with open(save_p, 'w') as fout:
        json.dump(dataset, fout)
        fout.close()

if __name__=='__main__':  # an example to read coco-format json using COCO API
    json_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/gt_coco'
    im_root = '/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images'

    with open(os.path.join(json_root, 'animal_pose_dataset.json'), 'r') as fin:
        dataset = json.load(fin)
        fin.close()

    # ----------------------------------
    # use COCO API to show an examlle
    from datasets.coco import COCO

    cocoGt = COCO(os.path.join(json_root, 'animal_pose_dataset.json'))
    cat_id = cocoGt.getCatIds(catNms=['cat'])
    print(cat_id)
    ann_ids = cocoGt.getAnnIds(catIds=cat_id)
    anns = cocoGt.loadAnns(ann_ids)
    NN = 3
    im_entry = cocoGt.loadImgs([anns[NN]['image_id']])
    I = Image.open(os.path.join(im_root, im_entry[0]['file_name'])).convert('RGB')
    plt.imshow(I)
    plt.axis('off')
    cocoGt.showAnns([anns[NN]], draw_bbox=True)
    plt.show()
    # ----------------------------------

