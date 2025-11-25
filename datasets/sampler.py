import os
# import pdb

import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

from collections import OrderedDict
import random


class MyBatchSampler(Sampler):
    def __init__(self, data_samples, chosen_kp_categories, KEYPOINT_TYPES, batch_type='one_class', batch_size=1, shuffle=False, drop_last=False, dataset_name='ANIMAL_POSE'):
        super(MyBatchSampler, self).__init__(data_samples)  # this line can be commented (optional)
        self.batch_type = batch_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.global_imagelist = OrderedDict()  # image list per kp_type
        self.perclass_imagelist = OrderedDict()
        self.perclass_imagenum = OrderedDict()

        self.nsamples = len(data_samples)
        for kp_type in KEYPOINT_TYPES:
            self.global_imagelist[kp_type] = []
        for i in range(self.nsamples):
            each_sample = data_samples[i]
            #----------------------------------------
            # this routine is used to handle different format of annotations in different datasets
            if dataset_name == 'ANIMAL_POSE':
                classid = each_sample['category']
                keypoints = each_sample['keypoints']
            elif dataset_name == 'CUB' or dataset_name == 'NAB' or dataset_name == 'AWA':
                classid = each_sample['classid']
                keypoints = each_sample['part']
            else:
                raise NotImplementedError
            # ----------------------------------------
            if classid not in self.perclass_imagelist:
                self.perclass_imagelist[classid] = OrderedDict()
                for kp_type in KEYPOINT_TYPES:
                    self.perclass_imagelist[classid][kp_type] = []
                self.perclass_imagenum[classid] = 0  # used for counting
            for j, kp_type in enumerate(KEYPOINT_TYPES):
                if keypoints[j][2] != 0:  # namely if image i has this keypoint category kp_type
                    self.global_imagelist[kp_type].append(i)  # append image i into the list
                    self.perclass_imagelist[classid][kp_type].append(i)
            self.perclass_imagenum[classid] += 1
        self.class_list = list(self.perclass_imagenum.keys())  # a list containing classid

        perclass_imagenum_value = np.array([self.perclass_imagenum[classid] for classid in self.class_list])
        self.perclass_prob = perclass_imagenum_value / np.sum(perclass_imagenum_value)


        order_fixed = True       # keypoint order
        if order_fixed == True:  # used when self.order_fixed = True
            if batch_type == 'mix_class':
                self.fixed_candidates_ids = self.acquire_image_candidates_ids(self.global_imagelist, chosen_kp_categories)  # a list
            else:  # 'one_class'
                self.fixed_candidates_ids_dict = self.acquire_image_candidates_ids2(chosen_kp_categories)  # a dict, each value (dictinary[classid]) is a list
                # convert set into list
                for classid in self.fixed_candidates_ids_dict:
                    self.fixed_candidates_ids_dict[classid] = list(self.fixed_candidates_ids_dict[classid])
        self.batch_cnt = 0  # record the total number of batches (each batch has batch_size images)

    def acquire_image_candidates_ids(self, imagelist: dict, chosen_kp_categories: list):
        candidates_ids = set()
        for kp_type in chosen_kp_categories:
            candidates_ids.update(set(imagelist[kp_type]))
        return list(candidates_ids)

    def acquire_image_candidates_ids2(self, chosen_kp_categories):
        candidates_dict = {}
        for classid in self.perclass_imagelist:
            candidates_dict[classid] = set()
            for kp_type in chosen_kp_categories:
                candidates_dict[classid].update(set(self.perclass_imagelist[classid][kp_type]))
        return candidates_dict



    def __iter__(self):
        if self.batch_type == 'mix_class':
            fixed_candidates_ids = deepcopy(self.fixed_candidates_ids)  # a list
            if self.shuffle:
                random.shuffle(fixed_candidates_ids)
        else:  # 'one_class'
            fixed_candidates_ids_dict = deepcopy(self.fixed_candidates_ids_dict)  # a dict
            if self.shuffle:
                for class_id in self.class_list:
                    random.shuffle(fixed_candidates_ids_dict[class_id])
            class_list_dynamic = deepcopy(self.class_list)

        while(1):
            id_list = []

            if self.batch_type == 'mix_class':
                if len(fixed_candidates_ids) < self.batch_size:
                    if self.drop_last or len(fixed_candidates_ids) == 0:
                        break
                    else:
                        for id in fixed_candidates_ids:
                            id_list.append(id)
                        break
                else:
                    for i in range(self.batch_size):
                        id_list.append(fixed_candidates_ids.pop(-1))  # pop from rear
            else:  # 'one_class'
                # sample a class
                if len(class_list_dynamic) == 0:
                    break

                # classid = np.random.choice(class_list_dynamic, size=1, replace=False, p=self.perclass_prob)  # sample a class
                classid = np.random.choice(class_list_dynamic, size=1)
                classid = classid[0]  # get value

                if len(fixed_candidates_ids_dict[classid]) < self.batch_size:
                    if self.drop_last or len(fixed_candidates_ids_dict[classid]) == 0:
                        class_list_dynamic.remove(classid)
                        continue
                    else:
                        for id in fixed_candidates_ids_dict[classid]:
                            id_list.append(id)
                        class_list_dynamic.remove(classid)

                else:
                    for i in range(self.batch_size):
                        id_list.append(fixed_candidates_ids_dict[classid].pop(-1))

            self.batch_cnt += 1
            yield id_list


    # not implemented
    # def __len__(self):
    #     return 0


class MetaSampler(Sampler):
    def __init__(self):
        super(MetaSampler, self).__init__()

    def __iter__(self):
        print('doing something')
        print('if here using <yield>, then iterator becomes generator and no need to return.')
        return 0

    def __next__(self):
        return 0

    def __len__(self):
        return 0