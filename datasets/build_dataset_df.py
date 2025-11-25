import os
import torch

import datasets.AnimalPoseDataset.animalpose_dataset as ANIMAL_POSE
import datasets.AwAPose.awa_pose_dataset as AWA
import datasets.CUB.cub_dataset as CUB
import datasets.NABird.nabird_dataset as NABIRD
import datasets.DeepFashion2.deepfashion2_dataset as DEEPFASHION2
from utils.utils import load_samples

import torchvision.transforms as transforms
import datasets.transforms as mytransforms

from datasets import sampler
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle

from datasets.text_prompts_input import generate_input_text_prompts
# from datasets.kp_names_mapping import get_mapped_kps_names
# from datasets.names_misc import get_obj_class_name_for_query_image, obj_class_name_preprocess


def build_dataset_json(cfg, phase='train'):
    # phase: 'train', 'val' (unseen species), 'test' (unseen species)

    if cfg.DATASET.TYPE == 'DEEPFASHION2':
        df_json_root = cfg.DATASET.DEEPFASHION2.JSON_ROOT
        train_json_files = ['deepfashion2_train_split2.json']
        val_json_files = ['deepfashion2_val_split2.json']

        use_hdf5 = cfg.DATASET.DEEPFASHION2.HDF5
        train_image_path = cfg.DATASET.DEEPFASHION2.TRAIN_IMAGE_PATH
        val_image_path = cfg.DATASET.DEEPFASHION2.VAL_IMAGE_PATH
        train_image_path_dict = {'hdf5': use_hdf5, 'status': 'train', 'path': train_image_path}
        val_image_path_dict = {'hdf5': use_hdf5, 'status': 'val', 'path': val_image_path}

        if phase == 'train':
            json_files = [os.path.join(df_json_root, f) for f in train_json_files]
            image_path_dict = train_image_path_dict
        elif phase == 'val' or phase == 'test':
            json_files = [os.path.join(df_json_root, f) for f in val_json_files]
            image_path_dict = val_image_path_dict
    else:
        raise NotImplementedError

    return json_files, image_path_dict

def build_dataset_meta(cfg):
    dataset_meta = {}

    if cfg.DATASET.TYPE == 'DEEPFASHION2':
        horizontal_swap_keypoints = DEEPFASHION2.horizontal_swap_keypoints
        get_auxiliary_paths = DEEPFASHION2.get_auxiliary_paths
        Dataset = DEEPFASHION2.DFDataset
        # hdf5_images_path = cfg.DATASET.DEEPFASHION2.HDF5_IMAGES_PATH
        saliency_maps_root = cfg.DATASET.DEEPFASHION2.SALIENCY_MAPS_ROOT
        # KEYPOINT_TYPES = DEEPFASHION2.KEYPOINT_TYPES
        EpisodeGenerator = DEEPFASHION2.EpisodeGenerator
    else:
        raise NotImplementedError

    dataset_meta['horizontal_swap_keypoints'] = horizontal_swap_keypoints
    dataset_meta['get_auxiliary_paths'] = get_auxiliary_paths
    dataset_meta['Dataset'] = Dataset
    # dataset_meta['hdf5_images_path'] = hdf5_images_path
    dataset_meta['saliency_maps_root'] = saliency_maps_root
    # dataset_meta['KEYPOINT_TYPES'] = KEYPOINT_TYPES  # for DF2, keypoint types are dynamically determined by category
    dataset_meta['EpisodeGenerator'] = EpisodeGenerator

    return dataset_meta


class build_episode_loader_df(object):
    def __init__(self, cfg, k_shot, m_query, least_s_kp_num, least_q_kp_num, episode_type='one_class', phase='train', **kwargs):
        # phase: 'train', 'val' (unseen species), 'test' (unseen species)
        json_files, image_path_dict = build_dataset_json(cfg, phase)
        dataset_meta = build_dataset_meta(cfg)

        episode_generator_list = []
        order_fixed = True  # True
        for f in json_files:
            episode_generator = dataset_meta['EpisodeGenerator'](f, K_shot=k_shot, M_queries=m_query,
                vis_requirement='partial_visible', least_support_kps_num=least_s_kp_num, least_query_kps_num=least_q_kp_num, episode_type=episode_type)  # partial_visible  full_visible
            episode_generator_list.append(episode_generator)
            # json_file_name = os.path.basename(f)
            # print('{}, Number of training images: {} / valid: {}'.format(json_file_name, len(episode_generator.samples), episode_generator.num_valid_image))

        self.episode_generator_list = episode_generator_list  # a list which contains one or multiple episode_generator
        self.dataset_meta = dataset_meta
        self.cfg = cfg
        self.episode_generator_chosen = None
        self.phase = phase
        self.n_way = 0  # number of support keypoints is dynamically determined by sampled clothes category in deepfashion2
        self.k_shot = k_shot
        self.m_query = m_query

        self.image_path_dict = image_path_dict

        self.sample_failure_cnt = 0
        self.sample_failure_cnt2 = 0

        # text prompts related
        self.num_text_per_obj = 0  #self.cfg.TRAIN.TEXT_PROMPT_SETTING.OBJ_TEXT if self.phase == 'train' else self.cfg.TEST.TEXT_PROMPT_SETTING.OBJ_TEXT
        # self.num_text_per_obj_dropout = True if (self.phase == 'train') and (self.cfg.TRAIN.TEXT_PROMPT_SETTING.DROPOUT_OBJ_TEXT==True) else False
        self.num_text_per_kp = 0  #self.cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT if self.phase == 'train' else self.cfg.TEST.TEXT_PROMPT_SETTING.NUM_TEXT
        # interpolated textual prompts
        self.generate_interpolated_texts = self.cfg.DATASET.GENERATE_INTERPOLATED_TEXTS
        if (self.generate_interpolated_texts == True) and (self.num_text_per_kp > 0) and (self.phase=='train'):
            raise NotImplementedError
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
            using_multiple_episodes = True
            num_multiple_episode = len(self.episode_generator_list)
            # prob = np.array([len(self.episode_generator_list[i].samples) for i in range(num_multiple_episode)])
            # prob = prob / np.sum(prob)
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
                    # mytransforms.RandomApply(mytransforms.HFlip(swap=dataset_meta['horizontal_swap_keypoints']), p=0.5),  # 0.5
                    mytransforms.RandomApply(mytransforms.RandomRotation(max_rotate_degree=15), p=0.25),  # 0.25
                    # mytransforms.RelativeResize((0.75, 1.25)),
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
                category_id = episode_generator.supports[0]['category_id']
                dataset_meta['KEYPOINT_TYPES'] = DEEPFASHION2.get_keypoints(category_id)  # record
                if generate_interpolated_kps == True:
                    # DF2 path_mode: 'random', 'neighbor_default'
                    assert auxiliary_path_mode in ['random', 'neighbor_default']
                    auxiliary_paths = dataset_meta['get_auxiliary_paths'](path_mode=auxiliary_path_mode,
                                                                          support_keypoint_categories=episode_generator.support_kp_categories,
                                                                          num_random_paths=num_random_paths,
                                                                          category_id=category_id
                                                                          )
                else:
                    auxiliary_paths = []
            else:
                # interpolated visual prompts
                generate_interpolated_kps = False
                interpolation_knots = None
                auxiliary_paths = []

            support_dataset = dataset_meta['Dataset'](episode_generator.supports,
                                                episode_generator.support_kp_categories,
                                                using_auxiliary_keypoints=generate_interpolated_kps,
                                                interpolation_knots=interpolation_knots,
                                                interpolation_mode=3,
                                                auxiliary_path=auxiliary_paths,
                                                images_path_dict=self.image_path_dict,
                                                saliency_maps_root=dataset_meta['saliency_maps_root'],
                                                output_saliency_map=output_saliency_map,
                                                preprocess=preprocess,
                                                input_transform=image_transform
                                                )
            support_loader = DataLoader(support_dataset, batch_size=self.k_shot, shuffle=False)
            support_loader_iter = iter(support_loader)
            (supports, support_labels, support_kp_mask, s_scale_trans, support_aux_kps, support_aux_kp_mask, support_saliency, s_bbox_origin, s_w_h_origin) = next(support_loader_iter)  # .next()

            #----------------------------------------------------------------------------------------
            # TODO: text prompt generation (Note object text depends on object class name of query image)
            obj_name = ''
            kps_names = ['']
            obj_texts, kps_texts = generate_input_text_prompts(obj_name, kps_names, self.num_text_per_obj, self.num_text_per_kp)  # a list, N_kps * T2
            obj_texts_mask = torch.ones(self.num_text_per_obj) * (self.num_text_per_obj > 0) # T1
            kps_texts_mask = torch.ones(len(episode_generator.support_kp_categories), self.num_text_per_kp) * (self.num_text_per_kp>0) # N x T2

            # interpolated text prompts generation (texts interpolated via LLM)
            if (self.generate_interpolated_texts == True) and (self.num_text_per_kp > 0) and (self.phase=='train'):
                raise NotImplementedError
            else:
                itpl_kps_texts = []  # an empty list
                # itpl_kps_texts_mask = torch.zeros(1, self.num_text_per_kp)  # N_aux x T2

                itpl_kps_texts_mask = torch.zeros(1, self.num_text_per_path)  # N_aux x T3
            # ----------------------------------------------------------------------------------------

            query_dataset = dataset_meta['Dataset'](episode_generator.queries,
                                              episode_generator.support_kp_categories,
                                              using_auxiliary_keypoints=generate_interpolated_kps,
                                              interpolation_knots=interpolation_knots,
                                              interpolation_mode=3,
                                              auxiliary_path=auxiliary_paths,
                                              images_path_dict=self.image_path_dict,
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
            union_kp_mask = (union_support_kp_mask + union_kps_texts_mask).long()  # N
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

