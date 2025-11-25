import os
import time
import datetime
import argparse
from yacs.config import CfgNode
import pprint
import logging
import numpy as np
import random
import copy
# import pdb

# import sys
# sys.path.append('.')  # append pwd into system path so that it could find python modules
# print(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.backends.cudnn as cudnn
# torch.autograd.set_detect_anomaly(True)
from tensorboardX import SummaryWriter

from utils.utils import list2str, load_samples, make_grid_images, image_normalize, mean_confidence_interval_multiple, mean_confidence_interval, AverageMeter, summarize_losses
from utils.sample_keypoints import PointSampler
# from utils.coco_eval_funs import compute_recall_ap
from datasets.kp_splits import train_test_kp_set
# from datasets.build_dataset_lw_stage1 import build_dataloader, build_dataset_meta
from datasets.build_dataset import build_episode_loader, build_dataset_meta, build_dataset_json
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from datasets.gt_itpl_texts import groundtruth_itpl_texts
from network.openkd_model import get_openkd_model, OpenKDModel
import network.clip_kd as clip
from core.loss_lw import HeatmapLoss, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from vis_diverse_texts import show_save_episode, save_predictions, save_heatmaps

import cv2
import json
import h5py
from PIL import Image
from datasets.coco import COCO
import torchvision.transforms as transforms
import datasets.transforms as mytransforms
from datasets.text_prompts_input import generate_input_text_prompts
from datasets.kp_names_mapping import get_mapped_kps_names
from difflib import SequenceMatcher

############################################################################################
## main call
############################################################################################
def update_config():
    parser = argparse.ArgumentParser(description='Open-prompted keypoint detection.')
    #./experiments/configs/openkd_autoname.yaml
    parser.add_argument('--autoname_keys', nargs='*', default=[])
    parser.add_argument('--autoname_labels', nargs='*', default=[])
    parser.add_argument('--cfg_file', type=str, default='./experiments/configs/openkd.yaml', help='config file')
    parser.add_argument('opts', help='see yaml config files for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = CfgNode.load_cfg(open(args.cfg_file))
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    print('-------------autoname key-labels start---------------')
    print(args.autoname_keys)
    print(args.autoname_labels)
    assert len(args.autoname_keys) == len(args.autoname_labels), 'keys/labels number should be same.'
    cfg.AUTONAME.KEYS = args.autoname_keys
    cfg.AUTONAME.LABELS = args.autoname_labels
    print('-------------autoname key-labels end---------------')

    return cfg, args

def create_loggers(cfg, cfg_str, init_logger=True, init_tb=True):
    output_dir = cfg.OUTPUT_DIR
    if os.path.exists(output_dir) == False:
        try:
            os.makedirs(output_dir)
        except:
            print('==>Cannot create folder at %s. Folder may already exist.'%(output_dir))
    output_model_dir = os.path.join(output_dir, 'model')
    output_log_dir = os.path.join(output_dir, 'log')
    output_tb_log_dir = os.path.join(output_dir, 'tb_log')
    for p in [output_model_dir, output_log_dir, output_tb_log_dir]:
        if os.path.exists(p) == False:
            try:
                os.makedirs(p)
            except:
                print('==>Cannot create folder at %s. Folder may already exist.'%(p))

    # set up logger
    if init_logger:
        logger_path = os.path.join(output_log_dir, cfg_str+'.log')
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(
            filename=str(logger_path),
            level=logging.INFO,
            format=head,
            # filemode='a',
            )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
    else:
        logger = None

    # set up tensorboard writer
    if init_tb:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        tb_writer_path = os.path.join(output_tb_log_dir, '{}-{}'.format(time_str, cfg_str))
        tb_writer = SummaryWriter(tb_writer_path)
    else:
        tb_writer = None

    return output_model_dir, logger, tb_writer

def get_align_kps_and_mask(support_kps, support_kp_mask, query_kps, query_kp_mask,
                           support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None):
    if support_aux_kps is None:
        align_kps = torch.cat([support_kps, query_kps], dim=1)                # S x (B1+B2) x N x 2
        align_kps_mask = torch.cat([support_kp_mask, query_kp_mask], dim=1)   # S x (B1+B2) x N
    else:
        s_kp_combined = torch.cat([support_kps, support_aux_kps], dim=-2)               # S x B1 x (N+T) x 2
        s_kp_mask_combined = torch.cat([support_kp_mask, support_aux_kp_mask], dim=-1)  # S x B1 x (N+T)
        q_kp_combined = torch.cat([query_kps, query_aux_kps], dim=-2)               # S x B2 x (N+T) x 2
        q_kp_mask_combined = torch.cat([query_kp_mask, query_aux_kp_mask], dim=-1)  # S x B2 x (N+T)
        align_kps = torch.cat([s_kp_combined, q_kp_combined], dim=1)                  # S x (B1+B2) x (N+T) x 2
        align_kps_mask = torch.cat([s_kp_mask_combined, q_kp_mask_combined], dim=1)   # S x (B1+B2) x (N+T)
    return align_kps, align_kps_mask

def recover_kps(kps, current_image_length, scale_trans):
    '''
    :param kps: B x M x 2 (range -1~1)
    :param current_image_length: 368
    :param scale_trans: B x 6, (scale, xoffset, yoffset, bbx_area, pad_xoffset, pad_yoffset)
    :return:
    '''
    B = kps.shape[0]
    kps = kps / 2 + 0.5 # 0~1, since our kp's range is -1~1 thus it needs to perform x/2 + 0.5
    kps *= current_image_length - 1
    kps += (scale_trans[:, 1:3]).view(B, 1, 2)
    kps /= (scale_trans[:, 0]).view(B, 1, 1)
    return kps

class EvalDataloader(object):
    def __init__(self, cfg, phase='test', **kwargs):
        # read diverse text prompts
        self.cfg = cfg
        self.dataset_type = cfg.DATASET.TYPE
        text_prompt_root = cfg.DATASET.DIVERSE_TEXT_EVAL_SETTING.ROOT
        self.llm         = cfg.DATASET.DIVERSE_TEXT_EVAL_SETTING.LLM
        K           = cfg.DATASET.DIVERSE_TEXT_EVAL_SETTING.NUM_TEXT_PROMPT
        version = "" if self.llm=='GPT' else "_vicuna"
        prompt_json = f"{text_prompt_root}/{self.dataset_type}_test_prompts_{K}_parse{version}.json"
        self.prompt_set = json.load(open(prompt_json, 'r'))
        print(f"==> Loaded diverse text prompts for dataset: {self.dataset_type}")
        if self.dataset_type == 'ANIMAL_POSE':  # 'cat', 'dog', 'cow', 'horse', 'sheep'
            self.prompt_set = self.prompt_set[cfg.DATASET.ANIMAL_POSE.UNSEEN_CLASS]
        self.num_text_prompts = len(self.prompt_set)

        # read COCO annotation
        json_files_list, obj_classes_list, images_path_dict = build_dataset_json(cfg, phase)
        dataset_meta = build_dataset_meta(cfg)
        self.cocoGT = COCO(json_files_list[0])
        print(f"==> Loaded COCO annotations for dataset: {self.dataset_type}")


        self.phase = phase
        self.images_path_dict = images_path_dict
        self.dataset_meta = dataset_meta

    def get_transforms(self):
        square_image_length = self.cfg.DATASET.SQUARE_IMAGE_LENGTH
        using_crop = True
        if using_crop:
            preprocess = mytransforms.Compose([
                mytransforms.RandomCrop(crop_bbox=False),
                mytransforms.Resize(longer_length=square_image_length),  # 384
                mytransforms.CenterPad(target_size=square_image_length),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
            ])
        else:
            preprocess = mytransforms.Compose([
                mytransforms.Resize(longer_length=square_image_length),  # 384
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

        return preprocess, image_transform

    def load_data(self, index):
        '''
        load data at index-th prompt
        :param index:
        :return:
        '''
        # TODO: 1. load text prompt
        anno_id, prompt_dict = self.prompt_set[index]
        diverse_text = prompt_dict['prompt']
        gt_obj_text  = prompt_dict['category']
        gt_kp_text   = prompt_dict['keypoints']  # a list of kp texts
        parsed_obj_text, parsed_kp_text = prompt_dict['parse']

        # TODO: 2. load GT kps and image data
        each_sample = self.cocoGT.anns[anno_id]
        keypoints = each_sample['keypoints']  # 1D list of [x, y, is_visible, ...]
        visible_bounds = each_sample['bbox']  # [xmin, ymin, w, h]

        image_id = each_sample['image_id']
        category_id = each_sample['category_id']
        image_entry = self.cocoGT.imgs[image_id]
        category_entry = self.cocoGT.cats[category_id]
        filename = image_entry['file_name']
        category = category_entry['name']
        assert gt_obj_text == category  # just for check

        if self.images_path_dict['hdf5'] is False:  # read from raw files
            image_path = os.path.join(self.images_path_dict['path'], filename)
            image = Image.open(image_path).convert('RGB')
            w, h = image.size  # PIL image
        else:  # read from hdf5
            hdf5_images_fin = h5py.File(self.images_path_dict['path'], 'r')
            key_for_image = filename
            jpeg_stream = hdf5_images_fin[key_for_image]
            image = cv2.imdecode(jpeg_stream[()], cv2.IMREAD_COLOR)  # cv2.IMREAD_UNCHANGED
            image = image[:, :, [2,1,0]]  # rgb
            image = Image.fromarray(image, mode='RGB')
            w, h = image.size
            hdf5_images_fin.close()  # close the hdf5 file
        w_h_origin = np.array([w, h])

        # TODO: 3. pre-process image data
        all_labels = np.array(keypoints, np.float64).reshape(-1, 3)  # N x 3
        bbox   = np.array([visible_bounds[0], visible_bounds[1], visible_bounds[2], visible_bounds[3]], np.float64)  # [xmin, ymin, w, h]
        # secure our bbox is within the image (some bboxes may be out of boundary)
        bbox[0], bbox[1] = max(bbox[0], 0), max(bbox[1], 0)
        bbox[2], bbox[3] = min(bbox[2], w - bbox[0]), min(bbox[3], h - bbox[1])
        bbox_origin = np.copy(bbox)  # [xmin, ymin, w, h]

        anno = {
            'keypoints': all_labels,
            'bbox': bbox
        }
        meta = {
            'scale': 1.0,
            'offset': np.array([0, 0], np.float64),
            'pad_offset': np.array([0, 0], np.float64),
            'valid_area': np.array([0, 0, w, h], np.float64),
            'hflip': False,  # may randomly flip and set it to be True in preprocess
        }
        preprocess, image_transform = self.get_transforms()
        image, anno, meta = preprocess(image, anno, meta)
        # re-set those invalid keypoint coordinates since they may be out of boundary
        all_labels_transformed = anno['keypoints']
        for i in range(all_labels_transformed.shape[0]):
            if all_labels_transformed[i, 2] == 0:  # invisible
                all_labels_transformed[i, :2] = 0  # set to be 0

        # scale, xoffset, yoffset, bbx_area, pad_xoffset, pad_yoffset
        scale_trans = np.array([meta['scale'], meta['offset'][0], meta['offset'][1], anno['bbox'][2] * anno['bbox'][3], meta['pad_offset'][0], meta['pad_offset'][1]])

        # extract the transformed keypoint labels relevant to our support keypoints
        label = np.zeros((len(gt_kp_text), 2))       # N x 2
        keypoint_mask = torch.ones(len(gt_kp_text))  # N
        KEYPOINT_TYPES = self.dataset_meta['KEYPOINT_TYPES']
        for i, kp_type in enumerate(gt_kp_text):
            kp_id = KEYPOINT_TYPES.index(kp_type)
            label[i, :] = all_labels_transformed[kp_id, :2]
            if all_labels_transformed[kp_id, 2] == 0:  # invisible
                keypoint_mask[i] = 0
            else:
                keypoint_mask[i] = 1

        image = image_transform(image)
        label = torch.tensor(label)
        scale_trans, bbox_origin, w_h_origin = torch.tensor(scale_trans), torch.tensor(bbox_origin), torch.tensor(w_h_origin)

        # TODO: 4. text prompt generation using texts parsed by LLM
        # if self.cfg.DATASET.DIVERSE_TEXT_EVAL_SETTING.USE_PARSED_OBJ_TEXT == True:
        #     if parsed_obj_text == 'n/a':
        #         text_prompts = parsed_kp_text
        #     else:
        #         _, text_prompts = generate_input_text_prompts(parsed_obj_text, parsed_kp_text, 0, 1)  # a list, N_kps * T
        # else:
        #     text_prompts = parsed_kp_text
        text_prompts = [diverse_text]  # TODO: modified for no parsing
        return text_prompts, image, label, keypoint_mask, scale_trans, bbox_origin, w_h_origin

def matching_texts(pred_texts, gt_texts):
    '''
    Compute the matching between two list of texts to determine which one matches which one
    :param pred_texts: a list of N1 texts
    :param gt_texts: a list of N2 texts
    :return: pred2gt, gt2pred
    '''
    N1 = len(pred_texts)
    N2 = len(gt_texts)
    iou_matrix = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            iou = SequenceMatcher(isjunk=None, a=pred_texts[i], b=gt_texts[j]).ratio()
            iou_matrix[i, j] = iou
    ids = np.argmax(iou_matrix, axis=1)  # N1
    pairs = list(zip(range(N1), ids))
    pred2gt = dict(pairs)
    gt2pred = {}
    for pred_id, gt_id in pred2gt.items():
        if gt2pred.get(gt_id) is None:
            gt2pred[gt_id] = []
        gt2pred[gt_id].append(pred_id)
    return pred2gt, gt2pred

def check_results(judges, gt2pred, gt_id_count_ranges, gt_kp_mask, pred_kp_mask):
    '''
    compute the accuracy for given gt keypoints
    :param judges:  N_pred x N_gt
    :param gt2pred: a dict
    :param gt_id_count_ranges: a list with length <= N_gt
    :param gt_kp_mask: N_gt
    :param pred_kp_mask: N_pred
    :return: accuracy
    '''
    cnt_correct = 0
    gt2pred_chosen = {}
    num_valid_kps = 0
    for gt_id in gt_id_count_ranges:
        if gt_kp_mask[gt_id] > 0:
            num_valid_kps += 1
            if gt2pred.get(gt_id) is not None:
                pred_ids = gt2pred[gt_id]
                for pred_id in pred_ids:
                    if (pred_kp_mask[pred_id] > 0) and (judges[pred_id, gt_id]):
                        cnt_correct += 1
                        gt2pred_chosen[gt_id] = pred_id
                        break  # break inner loop
    assert len(gt_id_count_ranges) > 0, 'it should give the gt id range to count'
    assert num_valid_kps > 0, 'there should be valid gt kps.'
    accuracy = cnt_correct / num_valid_kps
    return accuracy, gt2pred_chosen

def separate_base_novel_kps_for_texts(kp_texts, training_kp_category_set, testing_kp_category_set):
    base_ids = []
    novel_ids = []
    for i, t in enumerate(kp_texts):
        if t in training_kp_category_set:
            base_ids.append(i)
        elif t in testing_kp_category_set:
            novel_ids.append(i)
        else:
            raise NotImplementedError
    return base_ids, novel_ids

def compute_mean_ne(ne_matrix, gt2pred_chosen: dict):
    '''
    :param ne: N_pred x N_gt
    :param gt2pred_chosen: a dict
    :return: mean normalized error (ne)
    '''
    ne_sum = 0
    cnt = 0
    for gt_id, pred_id in gt2pred_chosen.items():
        ne = ne_matrix[pred_id, gt_id]
        ne_sum += ne
        cnt += 1
    if cnt > 0:
        mean_ne = ne_sum / cnt
    else:
        mean_ne = 0
    return mean_ne

def validate(cfg, model, test_episode_loader):
    print('==============testing start==============')
    torch.set_grad_enabled(False)  # disable grad computation
    # model.eval()

    square_image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH  # 384
    pck_thresh_bbx = np.array([0.10])  # np.array([0.10, 0.15])  # np.linspace(0, 1, 101)
    pck_thresh_img = np.array([0.06])  # np.array([0.06, 0.10])  # 0.06 * 384 = 23.04 pixels (23 pixels)
    pck_thresh_type = 'bbx'  # 'bbx' or 'img'
    if pck_thresh_type == 'bbx':  # == 'bbx'
        pck_thresh = pck_thresh_bbx
    else:  # == 'img'
        pck_thresh = pck_thresh_img
    tps, fps = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
    acc_list = [[] for _ in range(len(pck_thresh))]
    acc_list_base, acc_list_novel = copy.deepcopy(acc_list), copy.deepcopy(acc_list)
    ne_list = []  # normalized error
    ne_list_base, ne_list_novel = [], []

    episode_i = 0
    num_test_episodes = test_episode_loader.num_text_prompts

    multi_group_supervision = cfg.LOSS.MULTI_GROUP_SUPERVISION  # True or False
    fusing_operation = cfg.LOSS.OBJ_KP_HEATMAP_FUSION  # 'avg' or 'prod'

    training_kp_category_set, testing_kp_category_set, \
    least_s_kp_num, least_q_kp_num, least_s_kp_num2, least_q_kp_num2 = train_test_kp_set(cfg.DATASET.TYPE)
    while episode_i < num_test_episodes:
        # roll-out an episode
        # text_prompts: a list of N_t texts
        # queries: 3 x H x W
        # query_kp_mask: N
        # query_scale_trans: 6
        # query_bbx_origin: 4
        # query_w_h_origin: 2
        text_prompts, queries, query_labels, query_kp_mask, query_scale_trans, query_bbx_origin, query_w_h_origin = \
        test_episode_loader.load_data(index=episode_i)
        anno_id, prompt_dict = test_episode_loader.prompt_set[episode_i]

        parsed_kp_texts = prompt_dict['parse'][1]
        gt_kp_texts  = prompt_dict['keypoints']
        # pred2gt, gt2pred = matching_texts(parsed_kp_texts, gt_kp_texts)
        pred2gt, gt2pred = matching_texts(text_prompts, gt_kp_texts)  # TODO: modified for no parsing

        queries = queries.unsqueeze(0).unsqueeze(0)            # 1 x 1 x C x H x W
        query_labels = query_labels.unsqueeze(0).unsqueeze(0)  # 1 x 1 x N x 2
        query_kp_mask = query_kp_mask.unsqueeze(0).unsqueeze(0)        # 1 x 1 x N
        query_scale_trans = query_scale_trans.unsqueeze(0).unsqueeze(0)# 1 x 1 x 6
        query_bbx_origin = query_bbx_origin.unsqueeze(0).unsqueeze(0)  # 1 x 1 x 4
        query_w_h_origin = query_w_h_origin.unsqueeze(0).unsqueeze(0)  # 1 x 1 x 2

        # show_save_episode(None, None, None, queries[0], query_labels[0], query_kp_mask[0], None, episode_i,
        #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False,
        #                   save_root='output/episode_images', KEYPOINT_TYPES=test_episode_loader.dataset_meta['KEYPOINT_TYPES'],
        #                   support_kp_categories=gt_kp_texts)

        supports, queries = None, queries.cuda()  # S x B1 x C x H x W, S x B2 x C x H x W
        support_labels, query_labels = None, query_labels.cuda()  # S x B1 x N x 2, S x B2 x N x 2
        support_kp_mask = [[]]                 # S x B1 x N
        query_kp_mask = query_kp_mask.cuda()   # S x B2 x N
        obj_texts = [[]]                       # S x T1
        obj_texts_mask = None                  # S x T1
        kps_texts = [text_prompts]             # S x (N_t*T2)
        kps_texts_mask = torch.ones(1, len(text_prompts), 1, dtype=torch.int).cuda()  # S x N_t x T2

        if len(parsed_kp_texts)==0 or len(gt_kp_texts)==0:
            print(f'prompt error at episode {episode_i}')
            episode_i += 1
            continue  # skip

        # # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
        # union_support_kp_mask = torch.sum(support_kp_mask, dim=1) > 0  # S x N
        # # compute the valid query keypoints, using broadcast
        # valid_kp_mask = (query_kp_mask * union_support_kp_mask.unsqueeze(1))  # S x B2 x N
        # num_valid_kps = torch.sum(valid_kp_mask.flatten(1), dim=-1)   # S, valid kps per episode
        # num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=-1)  # S x B2

        # TODO: 5. feed model with diverse text prompts
        outputs = model(supports, queries, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'obj': tensor, 'text': tensor, 'image': tensor}
            support_kp_mask[0],
            kps_texts_mask[0],
            multi_group_supervision,
            fusing_operation
        )
        heatmaps_predict = heatmaps_fused  # B2 x N_t x h x w

        # multiple episode images into one (S=1)
        query_labels = query_labels[0]                                # B2 x N x 2
        # valid_kp_mask = query_kp_mask[0] * (fused_mask_sum>0).long()  # B2 x N
        query_kp_mask = query_kp_mask[0]                              # B2 x N
        query_scale_trans = query_scale_trans[0]                      # B2 x 6
        query_bbx_origin = query_bbx_origin[0]                        # B2 x 4
        query_w_h_origin = query_w_h_origin[0]                        # B2 x 2

        # coordinates decoding
        B2, N_t = heatmaps_predict.shape[:2]  # Note N_t may not be equal to GT number of keypoints due to text parsing
        N = query_kp_mask.shape[-1]
        if cfg.LOSS.TYPE == 'direct_coord':  # no need to decode
            predictions = heatmaps_predict
        else:
            H, W = heatmaps_predict.shape[2:]
            predict_score, predict_grids = torch.max(heatmaps_predict.reshape(B2, N_t, -1), 2)  # B2 x N_t
            predict_gridxy = torch.FloatTensor(B2, N_t, 2).cuda()
            predict_gridxy[:, :, 0] = predict_grids % W  # grid x
            predict_gridxy[:, :, 1] = predict_grids // H  # grid y
            # 'MSE', 'cross-entropy'
            predictions = ((predict_gridxy + 0.5) / H - 0.5) * 2  # B2 x N_t x 2

        predictions = predictions * (fused_mask_sum>0).long().view(1, N_t, 1)
        query_labels = query_labels * query_kp_mask.view(B2, N, 1)

        predictions = predictions.cpu().detach()    # B2 x N_t x 2
        query_labels = query_labels.cpu().detach()  # B2 x N x 2
        # valid_kp_mask = valid_kp_mask.cpu().detach()

        # # ----------------------------------------------------------------------------
        # # 1) save predicted keypoints
        # B1 = 0 if (supports is None) or (len(supports[0].shape) != 4) else supports[0].shape[0]  # Judge whether it is zero-shot or not
        # queries = queries[0].cpu().detach()           # B2 x C x H x W
        # query_kp_mask = query_kp_mask.cpu().detach()  # B2 x N
        # pred_kp_mask = (fused_mask_sum > 0).long().cpu().detach()  # 1 x N_t
        #
        # save_predictions(None, None, None, queries, query_labels, query_kp_mask, predictions,
        #                  test_episode_loader, episode_i,
        #                  KEYPOINT_TYPES=test_episode_loader.dataset_meta['KEYPOINT_TYPES'],limbs=[],
        #                  anno_id=anno_id, prompt_dict=prompt_dict, pred2gt=pred2gt, pred_kp_mask=pred_kp_mask
        #                  )
        #
        # # 2) save multi-group heatmaps / fused heatmaps
        # predict_heatmaps_dict = predict_heatmaps_list[0]  # a dict: {'obj': [], 'text': [], 'image': []}
        # draw_multi_group_heatmaps = []  # model.dict2list(predict_heatmaps_dict)  # a list of G heatmaps, each is B2 x N x h x w
        # draw_fused_heatmaps = heatmaps_fused.cpu().detach()    # B2 x N x h x w
        # draw_multi_group_heatmaps = list(map(lambda x: x.cpu().detach(), draw_multi_group_heatmaps))
        #
        # # ``queries" are query images, with size of B2 x C x H x W
        # save_heatmaps(queries, query_labels, query_kp_mask, predictions, draw_multi_group_heatmaps, draw_fused_heatmaps,
        #               test_episode_loader, episode_i,
        #               anno_id=anno_id, prompt_dict=prompt_dict, pred2gt=pred2gt, pred_kp_mask=pred_kp_mask
        #               )
        # # ----------------------------------------------------------------------------

        # TODO: 6. evaluation separately by base keypoints, novel keypoints, and combined
        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, query_scale_trans)
        query_labels_o = recover_kps(query_labels, square_image_length, query_scale_trans)

        # B2 x N_t x N
        square_diff = torch.sum((predictions_o.unsqueeze(-2) - query_labels_o.unsqueeze(-3)) ** 2, dim=-1).cpu().detach().numpy()
        if pck_thresh_type == 'bbx':
            longer_edge = np.max(query_bbx_origin[:, [2, 3]].numpy(), axis=1)  # B2, query_bbx_origin's format xmin, ymin, w, h
        else:  # == 'img'
            longer_edge = np.max(query_w_h_origin.numpy(), axis=1)
        longer_edge = longer_edge.reshape(-1, 1)  # B2 x 1

        gt_kp_mask = query_kp_mask.cpu().numpy()  # B2 x N
        pred_kp_mask = (fused_mask_sum>0).long().cpu().numpy()  # 1 x N_t

        square_diff = square_diff[0]    # N_t x N
        gt_kp_mask = gt_kp_mask[0]      # N
        pred_kp_mask = pred_kp_mask[0]  # N_t
        query_w_h_origin = query_w_h_origin[0]  # 2

        for ind, thr in enumerate(pck_thresh):
            judges = (square_diff <= (thr * longer_edge) ** 2)  # N_t x N

            # cnt_gt = (judges.sum(-2) > 0) * gt_kp_mask  # B2 x N
            # cnt_gt = cnt_gt.sum(-1)  # B2
            # cnt_pred = (judges.sum(-1) > 0) * pred_kp_mask  # B2 x N_t
            # cnt_pred = cnt_pred.sum(-1)  # B2
            # cnt_correct = np.minimum(cnt_gt, cnt_pred)  # B2
            # acc_cur = cnt_correct.sum() / gt_kp_mask.sum()
            # acc_list[ind].append(acc_cur)

            combined = list(range(len(gt_kp_texts)))
            base_ids, novel_ids = separate_base_novel_kps_for_texts(gt_kp_texts, training_kp_category_set, testing_kp_category_set)
            # if (len(combined) == 0) or (len(base_ids) == 0) or (len(novel_ids) == 0):
            #     print('check')
            # if episode_i in [11, 18]:
            #     pass
            acc_cur, gt2pred_chosen = check_results(judges, gt2pred, combined, gt_kp_mask, pred_kp_mask)
            acc_list[ind].append(acc_cur)
            if len(base_ids) > 0:
                acc_cur_base , gt2pred_chosen_base  = check_results(judges, gt2pred, base_ids, gt_kp_mask, pred_kp_mask)
                acc_list_base[ind].append(acc_cur_base)
            else:
                gt2pred_chosen_base = {}
            if len(novel_ids) > 0:
                acc_cur_novel, gt2pred_chosen_novel = check_results(judges, gt2pred, novel_ids, gt_kp_mask, pred_kp_mask)
                acc_list_novel[ind].append(acc_cur_novel)
            else:
                gt2pred_chosen_novel = {}

            if ind == 0:
                # compute mean normalized error in each episode
                ne_matrix = np.sqrt(square_diff) / np.max(query_w_h_origin.numpy(), axis=-1)
                if len(gt2pred_chosen) > 0:
                    ne_tmp = compute_mean_ne(ne_matrix, gt2pred_chosen)
                    ne_list.append(ne_tmp)
                if len(gt2pred_chosen_base) > 0:
                    ne_tmp_base  = compute_mean_ne(ne_matrix, gt2pred_chosen_base)
                    ne_list_base.append(ne_tmp_base)
                if len(gt2pred_chosen_novel) > 0:
                    ne_tmp_novel = compute_mean_ne(ne_matrix, gt2pred_chosen_novel)
                    ne_list_novel.append(ne_tmp_novel)

        if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)):
            acc_mean, interval = mean_confidence_interval_multiple(acc_list)
            acc_mean_base, interval_base = mean_confidence_interval_multiple(acc_list_base)
            acc_mean_novel, interval_novel = mean_confidence_interval_multiple(acc_list_novel)
            ne_mean, ne_interval = mean_confidence_interval(ne_list)
            ne_mean_base, ne_interval_base = mean_confidence_interval(ne_list_base)
            ne_mean_novel, ne_interval_novel = mean_confidence_interval(ne_list_novel)

            episode_curr = episode_i+1 if (episode_i == (num_test_episodes - 1)) else episode_i
            print_str = 'episode {}/{}, Acc {:.4f}, Int. {:.4f}, NE {:.6f}, Int. {:.6f}'.format(episode_curr, num_test_episodes, acc_mean[0], interval[0], ne_mean, ne_interval)
            print_str += '| Base acc {:.4f}, Int. {:.4f}, NE {:.6f}, Int. {:.6f}'.format(acc_mean_base[0], interval_base[0], ne_mean_base, ne_interval_base)
            print_str += '| Novel acc {:.4f}, Int. {:.4f}, NE {:.6f}, Int. {:.6f}'.format(acc_mean_novel[0], interval_novel[0], ne_mean_novel, ne_interval_novel)
            print_str += '| time: {}'.format(datetime.datetime.now())
            print(print_str)

        # increment in episode_i
        episode_i += 1

    print('==============testing end================')
    # model.train()
    torch.set_grad_enabled(True)  # enable grad computation

    return acc_mean, interval, ne_mean, ne_interval

def main():
    cfg, args = update_config()
    print(cfg)
    # print(pprint.pformat(cfg))

    manual_seed = cfg.MANUAL_SEED
    if manual_seed is not None:
        # cudnn.benchmark = False  # require to import torch.backends.cudnn as cudnn
        # cudnn.deterministic = True
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

    # two ways to control cfg_str, 2023.08.15
    if len(cfg.AUTONAME.KEYS) == 0:
        cfg_file_basename = os.path.basename(args.cfg_file)
        cfg_str = os.path.splitext(cfg_file_basename)[0]
    else:
        assert len(cfg.AUTONAME.LABELS) == len(cfg.AUTONAME.KEYS)
        cfg_str = ''
        for k, key_tmp in enumerate(cfg.AUTONAME.KEYS):
            label_tmp = cfg.AUTONAME.LABELS[k]
            value_tmp = eval('cfg.'+key_tmp)
            str_value = list2str(value_tmp) if isinstance(value_tmp, (list, tuple)) else str(value_tmp)
            if label_tmp == '':  # if label is an empty str, just continue
                continue
            cfg_str += (label_tmp+str_value)
    print('==>cfg_str: ', cfg_str)

    output_model_dir, logger, tb_writer = create_loggers(cfg, cfg_str, init_tb=False)
    # logger.info(cfg)

    print("==>Preparing model")
    openkd_model = get_openkd_model(cfg)
    # optimizer = get_optimizer(cfg, openkd_model)
    # if cfg.LOSS.TYPE in ['MSE', 'sigmoid-bce', 'cross-entropy']:  # supervised by GT heatmap
    #     loss_func = HeatmapLoss(cfg)
    # elif cfg.LOSS.TYPE == 'direct_coord':  # supervised by GT keypoints
    #     loss_func = DirectCoordLoss(cfg)
    # else:
    #     raise NotImplementedError

    checkpoint_file = os.path.join(output_model_dir, '%s.pth'%cfg_str)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        openkd_model.load_state_dict(checkpoint)
        print("==>Loaded checkpoint '{}'".format(checkpoint_file))

    if torch.cuda.is_available():
        openkd_model = openkd_model.cuda()
        # loss_func = loss_func.cuda()

    print("==>Preparing data")
    dataloader = EvalDataloader(cfg, phase='test')

    # train epochs
    meter = {
        'recall_stack': [0, 0],
        'recall_best': 0,
        'ne_stack': [1, 1],
        'ne_best': 1,
    }

    eval_cost = True
    openkd_model.set_cost_eval(eval_cost)

    print('==>Evaluate diverse text prompting on unseen species')
    eval_results = validate(cfg, openkd_model, dataloader)

    cost_base = openkd_model.get_cost_eval()  # record results
    print('cost_base:', cost_base)

    # print('cost_novel:', cost_novel)
    # print('Avg IT: %.6f sec/im, %.6f sec/kp'%((cost_base['IT1'] + cost_novel['IT1'])/2,
    #                                           (cost_base['IT2'] + cost_novel['IT2'])/2))

if __name__ == '__main__':
    main()