import os
import torch

import datasets.AnimalPoseDataset.animalpose_dataset as ANIMAL_POSE
import datasets.AwAPose.awa_pose_dataset as AWA
import datasets.CUB.cub_dataset as CUB
import datasets.NABird.nabird_dataset as NABIRD
from utils.utils import load_samples

import torchvision.transforms as transforms
import datasets.transforms as mytransforms

from datasets import sampler
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle

from datasets.text_prompts_input import generate_input_text_prompts
from datasets.kp_names_mapping import get_mapped_kps_names
from datasets.names_misc import get_obj_class_name_for_query_image, obj_class_name_preprocess


def build_dataset_json(cfg, phase='train'):
    # phase: 'train', 'val' (unseen species), 'test' (unseen species)

    if cfg.DATASET.TYPE == 'ANIMAL_POSE':
        json_root = cfg.DATASET.ANIMAL_POSE.JSON_ROOT

        classes = ['cat', 'cow', 'horse', 'sheep', 'dog']
        unseen_class = cfg.DATASET.ANIMAL_POSE.UNSEEN_CLASS
        seen_class = [class_i for class_i in classes if class_i != unseen_class]
        print('seen_class: ', seen_class)
        print('unseen_class: ', unseen_class)

        if phase == 'train':
            classes_tmp = seen_class
        elif phase == 'val' or phase == 'test':
            classes_tmp = [unseen_class]
        json_files_list= [os.path.join(json_root, 'animal_pose_dataset.json')]
        obj_classes_list = [classes_tmp]  # if i-th element is None, it means i-th json's all classes can be sampled.
        images_path_dict = {'hdf5': cfg.DATASET.ANIMAL_POSE.HDF5, 'path': cfg.DATASET.ANIMAL_POSE.IMAGE_ROOT}
    elif cfg.DATASET.TYPE == 'AWA':
        json_root = cfg.DATASET.AWA.JSON_ROOT

        training_json_files = ['AwAPose_split_train.json']
        testing_json_files  = ['AwAPose_split_test.json']
        val_json_files      = ['AwAPose_split_test.json']
        if phase == 'train':
            json_files_list = [os.path.join(json_root, f) for f in training_json_files]
        elif phase == 'test':
            json_files_list = [os.path.join(json_root, f) for f in testing_json_files]
        elif phase == 'val':
            json_files_list = [os.path.join(json_root, f) for f in val_json_files]
        obj_classes_list = [None]  # if i-th element is None, it means i-th json's all classes can be sampled.
        images_path_dict = {'hdf5': cfg.DATASET.AWA.HDF5, 'path': cfg.DATASET.AWA.IMAGE_ROOT}
    elif cfg.DATASET.TYPE == 'CUB':
        json_root = cfg.DATASET.CUB.JSON_ROOT

        training_json_files = ['cub_split_train.json']
        testing_json_files  = ['cub_split_test.json']
        val_json_files      = ['cub_split_val.json']
        if phase == 'train':
            json_files_list = [os.path.join(json_root, f) for f in training_json_files]
        elif phase == 'test':
            json_files_list = [os.path.join(json_root, f) for f in testing_json_files]
        elif phase == 'val':
            json_files_list = [os.path.join(json_root, f) for f in val_json_files]
        obj_classes_list = [None]  # if i-th element is None, it means i-th json's all classes can be sampled.
        images_path_dict = {'hdf5': cfg.DATASET.CUB.HDF5, 'path': cfg.DATASET.CUB.IMAGE_ROOT}
    elif cfg.DATASET.TYPE == 'NABIRD':
        json_root = cfg.DATASET.NABIRD.JSON_ROOT

        training_json_files = ['nabird_split_train.json']
        testing_json_files  = ['nabird_split_test.json']
        val_json_files      = ['nabird_split_val.json']
        if phase == 'train':
            json_files_list = [os.path.join(json_root, f) for f in training_json_files]
        elif phase == 'test':
            json_files_list = [os.path.join(json_root, f) for f in testing_json_files]
        elif phase == 'val':
            json_files_list = [os.path.join(json_root, f) for f in val_json_files]
        obj_classes_list = [None]  # if i-th element is None, it means i-th json's all classes can be sampled.
        images_path_dict = {'hdf5': cfg.DATASET.NABIRD.HDF5, 'path': cfg.DATASET.NABIRD.IMAGE_ROOT}
    else:
        raise NotImplementedError

    return json_files_list, obj_classes_list, images_path_dict

def build_dataset_meta(cfg):
    if cfg.DATASET.TYPE == 'ANIMAL_POSE':
        horizontal_swap_keypoints = ANIMAL_POSE.horizontal_swap_keypoints
        get_auxiliary_paths = ANIMAL_POSE.get_auxiliary_paths
        Dataset = ANIMAL_POSE.AnimalPoseDataset
        # hdf5_images_path = cfg.DATASET.ANIMAL_POSE.HDF5_IMAGES_PATH
        saliency_maps_root = cfg.DATASET.ANIMAL_POSE.SALIENCY_MAPS_ROOT
        KEYPOINT_TYPES = ANIMAL_POSE.KEYPOINT_TYPES
        EpisodeGenerator = ANIMAL_POSE.EpisodeGenerator
    elif cfg.DATASET.TYPE == 'AWA':
        horizontal_swap_keypoints = AWA.horizontal_swap_keypoints
        get_auxiliary_paths = AWA.get_auxiliary_paths
        Dataset = AWA.AwAPoseDataset
        # hdf5_images_path = cfg.DATASET.AWA.HDF5_IMAGES_PATH
        saliency_maps_root = cfg.DATASET.AWA.SALIENCY_MAPS_ROOT
        KEYPOINT_TYPES = AWA.KEYPOINT_TYPES
        EpisodeGenerator = AWA.EpisodeGenerator
    elif cfg.DATASET.TYPE == 'CUB':
        horizontal_swap_keypoints = CUB.horizontal_swap_keypoints
        get_auxiliary_paths = CUB.get_auxiliary_paths
        Dataset = CUB.CUBDataset
        # hdf5_images_path = cfg.DATASET.CUB.HDF5_IMAGES_PATH
        saliency_maps_root = cfg.DATASET.CUB.SALIENCY_MAPS_ROOT
        KEYPOINT_TYPES = CUB.KEYPOINT_TYPES
        EpisodeGenerator = CUB.EpisodeGenerator
    elif cfg.DATASET.TYPE == 'NABIRD':
        horizontal_swap_keypoints = NABIRD.horizontal_swap_keypoints
        get_auxiliary_paths = NABIRD.get_auxiliary_paths
        Dataset = NABIRD.NABirdDataset
        # hdf5_images_path = cfg.DATASET.NABIRD.HDF5_IMAGES_PATH
        saliency_maps_root = cfg.DATASET.NABIRD.SALIENCY_MAPS_ROOT
        KEYPOINT_TYPES = NABIRD.KEYPOINT_TYPES
        EpisodeGenerator = NABIRD.EpisodeGenerator
    else:
        raise NotImplementedError

    dataset_meta = {}
    dataset_meta['horizontal_swap_keypoints'] = horizontal_swap_keypoints
    dataset_meta['get_auxiliary_paths'] = get_auxiliary_paths
    dataset_meta['Dataset'] = Dataset
    # dataset_meta['hdf5_images_path'] = hdf5_images_path
    dataset_meta['saliency_maps_root'] = saliency_maps_root
    dataset_meta['KEYPOINT_TYPES'] = KEYPOINT_TYPES
    dataset_meta['EpisodeGenerator'] = EpisodeGenerator

    return dataset_meta

def get_itpl_texts_dict(cfg):
    dataset_type = cfg.DATASET.TYPE
    if dataset_type == 'ANIMAL_POSE':
        corpus_file = 'interpolated_animalpose'
    elif dataset_type == 'AWA':
        corpus_file = 'interpolated_awa'
    elif dataset_type == 'CUB':
        corpus_file = 'interpolated_cub'
    elif dataset_type == 'NABIRD':
        corpus_file = 'interpolated_nabird'
    else:
        raise NotImplementedError
    cfg_tmp = cfg.DATASET.ITPL_TEXT_CORPUS_SETTING
    corpus_root = cfg_tmp.ROOT
    N_repeat = cfg_tmp.NUM_REPEAT
    path = os.path.join(corpus_root, corpus_file + '_n{}.pkl'.format(N_repeat))
    with open(path, 'rb') as f:
        corpus_dict = pickle.load(f)
    f.close()

    itpl_type = cfg_tmp.TYPE
    k = cfg_tmp.NUM_TEXTS_PER_PATH
    cot = cfg_tmp.COT
    if  itpl_type == 'easy':
        variant = 'easy_raw_k%d_n%d'%(k, N_repeat)  # 'easy_raw_k1_n3', 'easy_raw_k3_n3'
    elif itpl_type == 'hard':
        cot = 'cot' if (cot == True) else 'raw'
        variant = 'hard_%s_k%d_n%d'%(cot, k, N_repeat)  # 'hard_raw_k3_n3', 'hard_cot_k3_n3'
    else:
        raise NotImplementedError
    itpl_texts_dict = corpus_dict[variant]
    return itpl_texts_dict, corpus_dict

class build_episode_loader(object):
    def __init__(self, cfg, support_kp_categories, n_way, k_shot, m_query, least_s_kp_num, least_q_kp_num, episode_type='one_class', phase='train', **kwargs):
        # phase: 'train', 'val' (unseen species), 'test' (unseen species)
        json_files_list, obj_classes_list, images_path_dict = build_dataset_json(cfg, phase)
        dataset_meta = build_dataset_meta(cfg)

        episode_generator_list = []
        order_fixed = True  # True
        if n_way > len(support_kp_categories):
            n_way = len(support_kp_categories)
        for i, f in enumerate(json_files_list):
            episode_generator = dataset_meta['EpisodeGenerator'](f, N_way=n_way, K_shot=k_shot, M_queries=m_query,
                kp_category_set=support_kp_categories, order_fixed=order_fixed, vis_requirement='partial_visible',
                least_support_kps_num=least_s_kp_num, least_query_kps_num=least_q_kp_num, episode_type=episode_type,
                obj_classes=obj_classes_list[i])  # partial_visible  full_visible
            episode_generator_list.append(episode_generator)
            json_file_name = os.path.basename(f)
            print('{}, Number of training images: {} / valid: {}'.format(json_file_name, episode_generator.total_instances, episode_generator.num_valid_instances))



        self.episode_generator_list = episode_generator_list  # a list which contains one or multiple episode_generator
        self.dataset_meta = dataset_meta
        self.cfg = cfg
        self.episode_generator_chosen = None
        self.phase = phase
        self.n_way = n_way
        self.k_shot = k_shot
        self.m_query = m_query
        self.images_path_dict = images_path_dict

        self.sample_failure_cnt = 0
        self.sample_failure_cnt2 = 0

        # text prompts related
        self.num_text_per_obj = self.cfg.TRAIN.TEXT_PROMPT_SETTING.OBJ_TEXT if self.phase == 'train' else self.cfg.TEST.TEXT_PROMPT_SETTING.OBJ_TEXT
        # self.num_text_per_obj_dropout = True if (self.phase == 'train') and (self.cfg.TRAIN.TEXT_PROMPT_SETTING.DROPOUT_OBJ_TEXT==True) else False
        self.num_text_per_kp = self.cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT if self.phase == 'train' else self.cfg.TEST.TEXT_PROMPT_SETTING.NUM_TEXT
        # interpolated textual prompts
        self.generate_interpolated_texts = self.cfg.DATASET.GENERATE_INTERPOLATED_TEXTS
        if (self.generate_interpolated_texts == True) and (self.num_text_per_kp > 0) and (self.phase=='train'):
            # itpl_text_path = 'datasets/text_corpus/itpl_text_animal.json'
            # with open(itpl_text_path, 'r') as fin:
            #     self.itpl_texts_dict = json.load(fin)
            #     fin.close()
            # first_path_key = list(self.itpl_texts_dict.keys())[0]
            # self.num_text_per_path = len(self.itpl_texts_dict[first_path_key])

            self.itpl_texts_dict, self.corpus_dict = get_itpl_texts_dict(cfg)
            first_path_key = list(self.itpl_texts_dict.keys())[0]
            self.num_text_gen_repeat = len(self.itpl_texts_dict[first_path_key])
            self.num_text_per_path = len(self.itpl_texts_dict[first_path_key][0])
            assert self.num_text_per_path > 0, 'the number of interpolated texts per path should > 0'
        else:
            self.generate_interpolated_texts = False
            self.itpl_texts_dict = None
            self.num_text_gen_repeat = 0
            self.num_text_per_path = 0

    def reset(self):
        self.sample_failure_cnt = 0
        self.sample_failure_cnt2 = 0

    def next(self, **kwargs):
        if len(self.episode_generator_list) > 0:  # if multiple_episode_generators is not empty
            num_multiple_episode = len(self.episode_generator_list)
            prob = np.array([self.episode_generator_list[i].num_valid_instances for i in range(num_multiple_episode)])
            prob = prob / np.sum(prob)
        else:
            raise NotImplementedError

        # roll-out an episode
        random_episode_ind = np.random.randint(0, num_multiple_episode, 1)
        # random_episode_ind = np.random.choice(range(0, num_multiple_episode), size=1, p=prob)
        # print('index: ', random_episode_ind)
        episode_generator = self.episode_generator_list[random_episode_ind[0]]

        self.episode_generator_chosen = episode_generator
        square_image_length = self.cfg.DATASET.SQUARE_IMAGE_LENGTH
        dataset_meta = self.dataset_meta
        output_saliency_map = self.cfg.DATASET.OUTPUT_SALIENCY_MAP

        while (True):
            sample_flag = episode_generator.episode_next()
            if False == sample_flag:
                self.sample_failure_cnt += 1
                if self.sample_failure_cnt % 500 == 0:
                    print('sample failure times: {}'.format(self.sample_failure_cnt))
                continue
            # print(episode_generator.support_kp_categories)

            # set up pre-process and image transforms
            if self.phase == 'train':
                # FSOD: support size 320 x 320, query size 1000 x 600, encoded feature is about 1/16
                # Openpose: image size 368 x 368, confidence map size 46 x 46
                # Mask-RCNN:
                preprocess = mytransforms.Compose([
                    # color transform
                    # mytransforms.RandomApply(mytransforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), p=0.8),
                    # mytransforms.RandomGrayscale(p=0.01),
                    # geometry transform
                    mytransforms.RandomApply(mytransforms.HFlip(swap=dataset_meta['horizontal_swap_keypoints']), p=0.5),  # 0.5
                    mytransforms.RandomApply(mytransforms.RandomRotation(max_rotate_degree=15), p=0.25),  # 0.25
                    mytransforms.RelativeResize((0.75, 1.25)),
                    mytransforms.RandomCrop(crop_bbox=False),
                    # mytransforms.RandomApply(mytransforms.RandomTranslation(), p=0.5),
                    mytransforms.Resize(longer_length=square_image_length),  # 368
                    mytransforms.CenterPad(target_size=square_image_length),
                    mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=False)
                ])
            else:  # 'test' or 'val'
                using_crop = True
                if using_crop:
                    preprocess = mytransforms.Compose([
                        mytransforms.RandomCrop(crop_bbox=False),
                        mytransforms.Resize(longer_length=square_image_length),  # 368
                        mytransforms.CenterPad(target_size=square_image_length),
                        mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                    ])
                else:
                    preprocess = mytransforms.Compose([
                        mytransforms.Resize(longer_length=square_image_length),  # 368
                        mytransforms.CenterPad(target_size=square_image_length),
                        mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
                    ])

            trunk = self.cfg.MODEL.ENCODER.TRUNK  # pre-trained model. Different pre-trained model has different normalized values.
            if 'RESNET' in trunk:  # pre-trained in ImageNet
                image_transform = transforms.Compose([
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    # transforms.RandomGrayscale(p=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            elif ('CLIP' in trunk) or ('BLIP' in trunk):  # CLIP pre-trained model (OPENAI uses private dataset for training)
                image_transform = transforms.Compose([
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    # transforms.RandomGrayscale(p=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                ])
            else:
                raise NotImplementedError

            if self.phase == 'train':
                # interpolated visual prompts
                generate_interpolated_kps = self.cfg.DATASET.GENERATE_INTERPOLATED_KPS
                interpolation_knots = np.array(self.cfg.DATASET.INTERPOLATION_KNOTS)

                # define a list containing the paths, each path is represented by kp index pair [index1, index2]
                # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
                num_random_paths = self.cfg.DATASET.NUM_RANDOM_PATHS  # only used when auxiliary_path_mode='random'
                auxiliary_path_mode = self.cfg.DATASET.AUXILIARY_PATH_MODE
                # path_mode:  'exhaust', 'predefined', 'random'
                auxiliary_paths = dataset_meta['get_auxiliary_paths'](path_mode=auxiliary_path_mode,
                                                                      support_keypoint_categories=episode_generator.support_kp_categories,
                                                                      num_random_paths=num_random_paths)
            else:
                # interpolated visual prompts
                generate_interpolated_kps = False
                interpolation_knots = None
                auxiliary_paths = []

            if self.k_shot >= 1:  # few-shot visual prompt
                support_dataset = dataset_meta['Dataset'](episode_generator.cocoGT,
                                                    episode_generator.supports,
                                                    episode_generator.support_kp_categories,
                                                    self.images_path_dict,
                                                    using_auxiliary_keypoints=generate_interpolated_kps,
                                                    interpolation_knots=interpolation_knots,
                                                    interpolation_mode=3,
                                                    auxiliary_path=auxiliary_paths,
                                                    saliency_maps_root=dataset_meta['saliency_maps_root'],
                                                    output_saliency_map=output_saliency_map,
                                                    preprocess=preprocess,
                                                    input_transform=image_transform
                                                    )
                support_loader = DataLoader(support_dataset, batch_size=self.k_shot, shuffle=False)
                support_loader_iter = iter(support_loader)
                (supports, support_labels, support_kp_mask, s_scale_trans, support_aux_kps, support_aux_kp_mask, support_saliency, s_bbox_origin, s_w_h_origin) = next(support_loader_iter) # support_loader_iter.next()
            else:  # zero-shot visual prompt (Fake. We use some zero to fill it.)
                t_zero = torch.tensor(0.)
                (supports, support_labels, support_kp_mask, s_scale_trans, support_aux_kps, support_aux_kp_mask, support_saliency, s_bbox_origin, s_w_h_origin) = [t_zero]*9
                # construct values (important to make code compatible between zero-shot and few-shot!)
                support_kp_mask = torch.zeros(1, self.n_way)
                support_aux_kp_mask = torch.zeros(1, len(auxiliary_paths) * len(self.cfg.DATASET.INTERPOLATION_KNOTS))

            #----------------------------------------------------------------------------------------
            # TODO: text prompt generation (Note object text depends on object class name of query image)
            obj_names = get_obj_class_name_for_query_image(episode_generator.queries, episode_generator.cocoGT)
            obj_name = obj_names[0]  # Use the name of first query image as the name for whole episode
            obj_name = obj_class_name_preprocess(self.cfg.DATASET.TYPE, obj_name)  # minor pre-processing to obj_name
            kps_names = get_mapped_kps_names(self.cfg.DATASET.TYPE, episode_generator.support_kp_categories, self.dataset_meta['KEYPOINT_TYPES'])  # a list, T1
            # if self.num_text_per_obj_dropout == False:
            #     num_text_per_obj = self.num_text_per_obj
            # else:
            #     num_text_per_obj = int(np.random.rand() > 0.5) * self.num_text_per_obj
            obj_texts, kps_texts = generate_input_text_prompts(obj_name, kps_names, self.num_text_per_obj, self.num_text_per_kp)  # a list, N_kps * T2
            obj_texts_mask = torch.ones(self.num_text_per_obj) * (self.num_text_per_obj > 0) # T1
            kps_texts_mask = torch.ones(len(episode_generator.support_kp_categories), self.num_text_per_kp) * (self.num_text_per_kp>0) # N x T2

            # interpolated text prompts generation (texts interpolated via LLM)
            if (self.generate_interpolated_texts == True) and (self.num_text_per_kp > 0) and (self.phase=='train'):
                # N_aux = len(auxiliary_paths) * len(interpolation_knots)
                # itpl_kps_texts = [""] * N_aux * self.num_text_per_kp    # a list, N_aux * T2. TODO: Waiting to implement
                # itpl_kps_texts_mask = torch.zeros(N_aux, self.num_text_per_kp)  # N_aux x T2. TODO: Waiting to implement

                N_path = len(auxiliary_paths)
                itpl_kps_texts = []  # N_path x T3 (T3 is the number of texts per path == Repeat*K texts per path)
                # itpl_kps_texts_mask = torch.ones(N_path, self.num_text_per_path)  # N_path x T3
                itpl_kps_texts_mask = torch.ones(N_path, self.num_text_gen_repeat * self.num_text_per_path)  # N_path x T3
                for per_path in auxiliary_paths:
                    # path_str = str(per_path)
                    # itpl_texts_per_path = self.itpl_texts_dict[path_str]  # retrieve the interpolated texts from dict
                    # itpl_kps_texts.extend(itpl_texts_per_path)

                    itpl_texts_per_path = []
                    text_pool_multi_repeat = self.itpl_texts_dict[tuple(per_path)]
                    list(map(itpl_texts_per_path.extend, text_pool_multi_repeat))  # list(map(...)) to combine a list of lists
                    itpl_kps_texts.extend(itpl_texts_per_path)
            else:
                itpl_kps_texts = []  # an empty list
                # itpl_kps_texts_mask = torch.zeros(1, self.num_text_per_kp)  # N_aux x T2

                itpl_kps_texts_mask = torch.zeros(1, self.num_text_per_path)  # N_aux x T3
            # ----------------------------------------------------------------------------------------

            query_dataset = dataset_meta['Dataset'](episode_generator.cocoGT,
                                              episode_generator.queries,
                                              episode_generator.support_kp_categories,
                                              self.images_path_dict,
                                              using_auxiliary_keypoints=generate_interpolated_kps,
                                              interpolation_knots=interpolation_knots,
                                              interpolation_mode=3,
                                              auxiliary_path=auxiliary_paths,
                                              saliency_maps_root=dataset_meta['saliency_maps_root'],
                                              output_saliency_map=output_saliency_map,
                                              preprocess=preprocess,
                                              input_transform=image_transform
                                              )
            query_loader = DataLoader(query_dataset, batch_size=self.m_query, shuffle=False)
            query_loader_iter = iter(query_loader)
            (queries, query_labels, query_kp_mask, q_scale_trans, query_aux_kps, query_aux_kp_mask, query_saliency, q_bbox_origin, q_w_h_origin) = next(query_loader_iter)  # .next()

            # check #1    there may exist some wrongly labeled images where the keypoints are outside the boundary
            if torch.any(support_labels > 1) or torch.any(support_labels < -1) or torch.any(query_labels > 1) or torch.any(query_labels < -1):
                self.sample_failure_cnt2 += 1
                if (self.sample_failure_cnt2 % 50 == 0):
                    print('count for wrongly labelled images: {}'.format(self.sample_failure_cnt2))
                continue             # skip current episode directly

            # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
            union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0 # N
            union_kps_texts_mask = torch.sum(kps_texts_mask, dim=-1) > 0  # N
            union_kp_mask = union_support_kp_mask + union_kps_texts_mask  # N
            # compute the valid query keypoints, using broadcast
            valid_kp_mask = (query_kp_mask * union_kp_mask.reshape(1, -1))  # B2 x N
            num_valid_kps = torch.sum(valid_kp_mask)

            # check #2
            if num_valid_kps ==  0:  # flip transform may lead zero intersecting keypoints between support and query, namely zero valid kps
                self.sample_failure_cnt2 += 1
                if (self.sample_failure_cnt2 % 100 == 0):
                    print('count for invalid episodes: {}'.format(self.sample_failure_cnt2))
                continue             # skip current episode directly

            break  # success

        # return three data tuples: (support data) & (query data) & (text prompts)
        return (supports, support_labels, support_kp_mask, s_scale_trans, support_aux_kps, support_aux_kp_mask, support_saliency, s_bbox_origin, s_w_h_origin), \
               (queries, query_labels, query_kp_mask, q_scale_trans, query_aux_kps, query_aux_kp_mask, query_saliency, q_bbox_origin, q_w_h_origin), \
               (obj_texts, kps_texts, obj_texts_mask, kps_texts_mask, itpl_kps_texts, itpl_kps_texts_mask)

    def next_multi_episodes(self, s=1, **kwargs):
        # roll out s episodes per time. s >= 1.
        supports_ = []
        support_labels_ = []
        support_kp_mask_ = []
        s_scale_trans_ = []
        support_aux_kps_ = []
        support_aux_kp_mask_ = []
        support_saliency_ = []
        s_bbox_origin_ = []
        s_w_h_origin_ = []

        queries_ = []
        query_labels_ = []
        query_kp_mask_ = []
        q_scale_trans_ = []
        query_aux_kps_ = []
        query_aux_kp_mask_ = []
        query_saliency_ = []
        q_bbox_origin_ = []
        q_w_h_origin_ = []

        obj_texts_ = []
        kps_texts_ = []
        obj_texts_mask_, kps_texts_mask_ = [], []
        itpl_kps_texts_ = []
        itpl_kps_texts_mask_ = []

        for i in range(int(s)):
            (supports, support_labels, support_kp_mask, s_scale_trans, support_aux_kps, support_aux_kp_mask, support_saliency, s_bbox_origin, s_w_h_origin), \
            (queries, query_labels, query_kp_mask, q_scale_trans, query_aux_kps, query_aux_kp_mask, query_saliency, q_bbox_origin, q_w_h_origin), \
            (obj_texts, kps_texts, obj_texts_mask, kps_texts_mask, itpl_kps_texts, itpl_kps_texts_mask) = self.next()
            supports_.append(supports)
            support_labels_.append(support_labels)
            support_kp_mask_.append(support_kp_mask)
            s_scale_trans_.append(s_scale_trans)
            support_aux_kps_.append(support_aux_kps)
            support_aux_kp_mask_.append(support_aux_kp_mask)
            support_saliency_.append(support_saliency)
            s_bbox_origin_.append(s_bbox_origin)
            s_w_h_origin_.append(s_w_h_origin)

            queries_.append(queries)
            query_labels_.append(query_labels)
            query_kp_mask_.append(query_kp_mask)
            q_scale_trans_.append(q_scale_trans)
            query_aux_kps_.append(query_aux_kps)
            query_aux_kp_mask_.append(query_aux_kp_mask)
            query_saliency_.append(query_saliency)
            q_bbox_origin_.append(q_bbox_origin)
            q_w_h_origin_.append(q_w_h_origin)

            obj_texts_.append(obj_texts)
            kps_texts_.append(kps_texts)
            obj_texts_mask_.append(obj_texts_mask)
            kps_texts_mask_.append(kps_texts_mask)
            itpl_kps_texts_.append(itpl_kps_texts)
            itpl_kps_texts_mask_.append(itpl_kps_texts_mask)

        supports_ = torch.stack(supports_, dim=0)
        support_labels_ = torch.stack(support_labels_, dim=0)
        support_kp_mask_ = torch.stack(support_kp_mask_, dim=0)
        s_scale_trans_ = torch.stack(s_scale_trans_, dim=0)
        support_aux_kps_ = torch.stack(support_aux_kps_, dim=0)
        support_aux_kp_mask_ = torch.stack(support_aux_kp_mask_, dim=0)
        support_saliency_ = torch.stack(support_saliency_, dim=0)
        s_bbox_origin_ = torch.stack(s_bbox_origin_, dim=0)
        s_w_h_origin_ = torch.stack(s_w_h_origin_, dim=0)

        queries_ = torch.stack(queries_, dim=0)
        query_labels_ = torch.stack(query_labels_, dim=0)
        query_kp_mask_ = torch.stack(query_kp_mask_, dim=0)
        q_scale_trans_ = torch.stack(q_scale_trans_, dim=0)
        query_aux_kps_ = torch.stack(query_aux_kps_, dim=0)
        query_aux_kp_mask_ = torch.stack(query_aux_kp_mask_, dim=0)
        query_saliency_ = torch.stack(query_saliency_, dim=0)
        q_bbox_origin_ = torch.stack(q_bbox_origin_, dim=0)
        q_w_h_origin_ = torch.stack(q_w_h_origin_, dim=0)

        obj_texts_mask_ = torch.stack(obj_texts_mask_, dim=0)
        kps_texts_mask_ = torch.stack(kps_texts_mask_, dim=0)
        itpl_kps_texts_mask_ = torch.stack(itpl_kps_texts_mask_, dim=0)

        return (supports_, support_labels_, support_kp_mask_, s_scale_trans_, support_aux_kps_, support_aux_kp_mask_, support_saliency_, s_bbox_origin_, s_w_h_origin_), \
               (queries_, query_labels_, query_kp_mask_, q_scale_trans_, query_aux_kps_, query_aux_kp_mask_, query_saliency_, q_bbox_origin_, q_w_h_origin_), \
               (obj_texts_, kps_texts_, obj_texts_mask_, kps_texts_mask_, itpl_kps_texts_, itpl_kps_texts_mask_)

