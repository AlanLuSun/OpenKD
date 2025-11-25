import os
import pickle
import time
import datetime
import argparse
from yacs.config import CfgNode
import pprint
import logging
import numpy as np
import random
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
from datasets.build_dataset import build_episode_loader, build_dataset_meta
from datasets.build_dataset_df import build_episode_loader_df
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from network.openkd_model import get_openkd_model, OpenKDModel
import network.clip_kd as clip
from core.loss_lw import HeatmapLoss, GM_GM_L2_Wrap, GMM_SmootherV2, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from vis import show_save_episode, save_predictions, save_heatmaps

from network.models_gridms2 import extract_representations, average_representations2

import cv2
import pickle
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

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
        os.makedirs(output_dir)
    output_model_dir = os.path.join(output_dir, 'model')
    output_log_dir = os.path.join(output_dir, 'log')
    output_tb_log_dir = os.path.join(output_dir, 'tb_log')
    for p in [output_model_dir, output_log_dir, output_tb_log_dir]:
        if os.path.exists(p) == False:
            os.makedirs(p)

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

def get_optimizer(cfg, model):
    optimizer_type = cfg.TRAIN.OPTIMIZER
    lr = cfg.TRAIN.LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    if  optimizer_type== 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError

    return optimizer

def train_episodes(cfg, model:torch.nn.Module, loss_func, optimizer, train_episode_loader, test_episode_loader, test_episode_loader2=None, writer=None, **kwargs):
    episode_i = 0
    generate_interpolated_kps = cfg.DATASET.GENERATE_INTERPOLATED_KPS
    # interpolation_knots    = cfg.DATASET.INTERPOLATION_KNOTS
    generate_interpolated_texts = cfg.DATASET.GENERATE_INTERPOLATED_TEXTS

    # training setting related
    T1 = cfg.TRAIN.TEXT_PROMPT_SETTING.OBJ_TEXT  # number of text per object
    T2 = cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT  # number of text per main keypoint
    T3 = train_episode_loader.num_text_per_path  # number of interpolated texts per path
    ENABLE_ITPL_TEXT = cfg.TRAIN.TEXT_PROMPT_SETTING.ENABLE_ITPL_TEXT
    B1 = cfg.TRAIN.NUM_TRAIN_SHOT
    B2 = cfg.TRAIN.NUM_TRAIN_QUERY
    ENABLE_ITPL_VISUAL = cfg.TRAIN.ENABLE_ITPL_VISUAL

    train_episode_loader.reset()  # clear sampling failure counters
    # if cfg.LOSS.DOMAIN_ALIGNMENT.TYPE != None:
    #     sample_neg_kps_type = cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.TYPE
    #     if sample_neg_kps_type is not None:
    #         negative_point_sampler = PointSampler(type=sample_neg_kps_type,
    #                                     num_kps=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.NUM_PER_IM,
    #                                     dist_thresh=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.DIST_THRESH,
    #                                     bbx_extend_ratio=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.BBX_EXTEND_RATIO,
    #                                     )
    #     else:
    #         negative_point_sampler = None
    while episode_i < cfg.TRAIN.NUM_EPISODES:
        if episode_i % 400 == 0 and episode_i > 0:
            eval_results = validate(cfg, model, test_episode_loader)
            recall = eval_results[0]
            ne = eval_results[2]
            if test_episode_loader2 != None:
                eval_results2 = validate(cfg, model, test_episode_loader2)
                recall2 = eval_results2[0]
                ne2 = eval_results2[2]
            if writer != None:
                # recall is a list which is corresponding to different thresholds
                writer.add_scalar('accuracy', recall[0], episode_i)
                if test_episode_loader2 != None:
                    writer.add_scalar('accuracy2', recall2[0], episode_i)
            # save model based on the configurations said in self.opts
            recall_stack = kwargs['meter']['recall_stack']
            recall_best = kwargs['meter']['recall_best']
            ne_stack = kwargs['meter']['ne_stack']
            ne_best = kwargs['meter']['ne_best']
            recall_stack[0] = recall_stack[1]
            if cfg.DATASET.TYPE in ['ANIMAL_POSE', 'AWA', 'CUB', 'NABIRD']:
                recall_stack[1] = (2*recall2[0] * recall[0])/(recall2[0] + recall[0] + 1e-9)  # recall2[0]
            else:  # 'DEEPFASHION2'
                recall_stack[1] = recall[0]
            avg_recall = recall_stack[1]  # np.mean(recall_stack)
            if avg_recall > recall_best:
                if episode_i >= 0:
                    torch.save(model.state_dict(), kwargs['checkpoint_file'])
                recall_best = avg_recall
                print('BEST: %s'%(recall_best))
            print('Curr: %s' % (avg_recall))
            kwargs['meter']['recall_stack'] = recall_stack
            kwargs['meter']['recall_best'] = recall_best

        (supports, support_labels, support_kp_mask, _, support_aux_kps, support_aux_kp_mask, support_saliency, _, _), \
        (queries, query_labels, query_kp_mask, _, query_aux_kps, query_aux_kp_mask, query_saliency, _, _), \
        (obj_texts, kps_texts, obj_texts_mask, kps_texts_mask, itpl_texts_pool, itpl_texts_pool_mask) = \
            train_episode_loader.next_multi_episodes(s=cfg.TRAIN.NUM_ROLL_OUT)

        # make_grid_images(supports[0], denormalize=True, save_path='grid_image_s.jpg')
        # make_grid_images(queries[0], denormalize=True, save_path='grid_image_q.jpg')
        ## 'exhaust', 'predefined'
        # save_episode_before_preprocess(episode_generator, episode_i, delete_old_files=False, draw_interpolated_kps=generate_interpolated_kps, interpolation_knots=interpolation_knots, interpolation_mode=self.opts['interpolation_mode'], path_mode='predefined')
        # show_save_episode(supports[0], support_labels[0], support_kp_mask[0], queries[0], query_labels[0], query_kp_mask[0], train_episode_loader.episode_generator_chosen, episode_i,
        #                   support_aux_kps[0], support_aux_kp_mask[0], query_aux_kps[0], query_aux_kp_mask[0], is_show=False, is_save=True, delete_old_files=False,
        #                   save_root='output/episode_images', KEYPOINT_TYPES=train_episode_loader.dataset_meta['KEYPOINT_TYPES'])
        # show_save_episode(supports[0], support_labels[0], support_kp_mask[0], queries[0], query_labels[0], query_kp_mask[0], train_episode_loader.episode_generator_chosen, episode_i,
        #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False,
        #                   save_root='output/episode_images', KEYPOINT_TYPES=train_episode_loader.dataset_meta['KEYPOINT_TYPES'])
        # print_weights(episode_generator.support_kp_mask)

        supports, queries = supports.cuda(), queries.cuda()  # S x B1 x C x H x W, S x B2 x C x H x W
        support_labels, query_labels = support_labels.float().cuda(), query_labels.float().cuda()  # S x B1 x N x 2, S x B2 x N x 2
        support_kp_mask = support_kp_mask.cuda()  # S x B1 x N
        query_kp_mask = query_kp_mask.cuda()      # S x B2 x N
        if generate_interpolated_kps:
            support_aux_kps = support_aux_kps.float().cuda()  # S x B1 x A x 2, A = (N_paths * N_knots), total number of auxiliary keypoints
            support_aux_kp_mask = support_aux_kp_mask.cuda()  # S x B1 x A
            query_aux_kps = query_aux_kps.float().cuda()      # S x B2 x A x 2
            query_aux_kp_mask = query_aux_kp_mask.cuda()      # S x B2 x A
        obj_texts_mask = obj_texts_mask.cuda()  # S x T1
        kps_texts_mask = kps_texts_mask.cuda()  # S x N x T2
        if generate_interpolated_texts:
            itpl_texts_pool_mask = itpl_texts_pool_mask.cuda()  # S x N_path x T3

        # # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
        # union_support_kp_mask = torch.sum(support_kp_mask, dim=1) > 0  # S x N
        # # compute the valid query keypoints, using broadcast
        # valid_kp_mask = (query_kp_mask * union_support_kp_mask.unsqueeze(1))  # S x B2 x N
        # num_valid_kps = torch.sum(valid_kp_mask.flatten(1), dim=-1)  # S, valid kps per episode
        # if generate_interpolated_kps:
        #     union_support_aux_kp_mask = torch.sum(support_aux_kp_mask, dim=1) > 0  # S x T
        #     valid_aux_kp_mask = query_aux_kp_mask * union_support_aux_kp_mask.unsqueeze(1)  # S x B2 x T
        #     num_valid_aux_kps = torch.sum(valid_aux_kp_mask.flatten(1), dim=-1)  # S
        #     # print(num_valid_kps+num_valid_aux_kps)

        # TODO: Training settings
        if (ENABLE_ITPL_TEXT == False) and (ENABLE_ITPL_VISUAL == False):  # 3 settings: text (main), visual (main), text (main)+visual(main)
            N = query_kp_mask.shape[-1]  # S x B2 x N
            outputs = model(supports, queries, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask,
                            num_main_kps=N,                # used for domain alignment
                            query_kps_=query_labels,       # used for domain alignment
                            query_kp_mask_=query_kp_mask,  # used for domain alignment
                            )
            predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
            loss_v_t_align, loss_v_v_align, loss_t_t_align = outputs[1], outputs[2], outputs[3]  # a scalar
            loss_v_t_itpl = None
            loss_align = summarize_losses([loss_v_t_align, loss_v_v_align, loss_t_t_align], cfg.LOSS.DOMAIN_ALIGNMENT.WEIGHT_ALIGN)
            loss_main = compute_openkd_heatmap_loss(cfg, model, loss_func, query_labels, query_kp_mask,
                                               predict_heatmaps_list, support_kp_mask, kps_texts_mask)
            loss_weights = cfg.LOSS.WEIGHT_MAIN_AUX
            loss = loss_main if (loss_align is None) else (loss_weights[0] * loss_main + loss_weights[2] * loss_align)
            loss_aux = None
        else:  # 5 settings
            assert B1 > 0 or T2 > 0, 'At least have one type of prompts.'
            N, A = query_kp_mask.shape[-1], query_aux_kp_mask.shape[-1]  # S x B2 x N, S x B2 x A
            # N, N_path = kps_texts_mask.shape[1], itpl_texts_pool_mask.shape[1]  # S x N x T2, S x N_path x T3
            S = cfg.TRAIN.NUM_ROLL_OUT
            if (T2 == 0) and (B1 > 0):  # setting: visual (main+aux)
                # the kps_texts and itpl_texts_pool are empty list in this case!
                assert ENABLE_ITPL_VISUAL == True, 'ENABLE_ITPL_VISUAL should be True. Info:\n{}'.format(cfg.TRAIN)
                s_kp_combined = torch.cat([support_labels, support_aux_kps], dim=-2)            # S x B1 x (N+A) x 2
                s_kp_mask_combined = torch.cat([support_kp_mask, support_aux_kp_mask], dim=-1)  # S x B1 x (N+A)
            elif (T2 > 0) and (B1 == 0):  # setting: textual (main+aux)
                # the visual prompts are none in this case!
                assert ENABLE_ITPL_TEXT==True, 'ENABLE_ITPL_TEXT should be True. Info:\n{}'.format(cfg.TRAIN)
                s_kp_combined = None
                s_kp_mask_combined = None
            else:
                assert (T2 > 0) and (B1 > 0), 'Both texual and visual prompts are reuiqred. Interpolated kps or texts are required, too.'
                if (ENABLE_ITPL_TEXT==False) and (ENABLE_ITPL_VISUAL==True): # setting: textual(main) + visual(main+aux)
                    itpl_texts_pool = [[]] * S    # set empty list, only padding, no need to infer interpolated texts
                    itpl_texts_pool_mask = None
                    s_kp_combined = torch.cat([support_labels, support_aux_kps], dim=-2)            # S x B1 x (N+A) x 2
                    s_kp_mask_combined = torch.cat([support_kp_mask, support_aux_kp_mask], dim=-1)  # S x B1 x (N+A)
                elif (ENABLE_ITPL_TEXT==True) and (ENABLE_ITPL_VISUAL==False):# setting: textual(main+aux) + visual(main)
                    s_kp_combined = torch.cat([support_labels, support_aux_kps], dim=-2)  # S x B1 x (N+A) x 2
                    support_aux_kp_mask *= 0  # set 0, masking
                    s_kp_mask_combined = torch.cat([support_kp_mask, support_aux_kp_mask], dim=-1)  # S x B1 x (N+A)
                elif (ENABLE_ITPL_TEXT==True) and (ENABLE_ITPL_VISUAL==True):# setting: textual(main+aux) + visual(main+aux)
                    s_kp_combined = torch.cat([support_labels, support_aux_kps], dim=-2)            # S x B1 x (N+A) x 2
                    s_kp_mask_combined = torch.cat([support_kp_mask, support_aux_kp_mask], dim=-1)  # S x B1 x (N+A)
                else:
                    raise NotImplementedError

            q_kp_combined = torch.cat([query_labels, query_aux_kps], dim=-2)            # S x B2 x (N+A) x 2
            q_kp_mask_combined = torch.cat([query_kp_mask, query_aux_kp_mask], dim=-1)  # S x B2 x (N+A)
            outputs = model(supports, queries, s_kp_combined, s_kp_mask_combined, obj_texts, obj_texts_mask,
                            kps_texts, kps_texts_mask, itpl_texts_pool, itpl_texts_pool_mask,
                            num_main_kps=N,                     # used for domain alignment
                            query_kps_=q_kp_combined,           # used for domain alignment
                            query_kp_mask_=q_kp_mask_combined,  # used for domain alignment
                            )
            predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
            loss_v_t_align, loss_v_v_align, loss_t_t_align = outputs[1], outputs[2], outputs[3]  # a scalar
            loss_v_t_itpl = outputs[4]  # a scalar
            loss_align = summarize_losses([loss_v_t_align, loss_v_v_align, loss_t_t_align, loss_v_t_itpl], cfg.LOSS.DOMAIN_ALIGNMENT.WEIGHT_ALIGN)
            itpl_kps_texts_info = model.get_complement_itpl_kps_texts_info()
            itpl_kps_texts_mask = itpl_kps_texts_info['mask']
            main_heatmaps_list, aux_heatmaps_list = split_main_aux_heatmaps(predict_heatmaps_list, num_main_kps=N, num_aux_kps=A)
            loss_main = compute_openkd_heatmap_loss(cfg, model, loss_func, query_labels, query_kp_mask,
                                                    main_heatmaps_list, support_kp_mask, kps_texts_mask)
            loss_aux = compute_openkd_heatmap_loss(cfg, model, loss_func, query_aux_kps, query_aux_kp_mask,
                                                   aux_heatmaps_list, support_aux_kp_mask, itpl_kps_texts_mask)
            loss_weights = cfg.LOSS.WEIGHT_MAIN_AUX
            loss = loss_weights[0] * loss_main + loss_weights[1] * loss_aux
            if loss_align is not None:
                loss += loss_weights[2] * loss_align

        if np.isnan(loss.cpu().detach().numpy()):
            print('error nan in loss similarity')
            continue
        if np.isinf(loss.cpu().detach().numpy()):
            print('error inf in loss similarity')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode_i) % 8 == 0:
            msg = 'episode: {}, loss: {:.5f}'.format(episode_i, loss.item())
            if loss_main is not None:
                msg += '/main: {:.5f}'.format(loss_main.item())
            if loss_aux is not None:
                msg += ', aux: {:.5f}'.format(loss_aux.item())
            if loss_align is not None:
                msg += ', align: {:.5f}'.format(loss_align.item())
            if loss_v_t_align is not None:
                msg += ', v_t: {:.5f}'.format(loss_v_t_align.item())
            if loss_v_v_align is not None:
                msg += ', v_v: {:.5f}'.format(loss_v_v_align.item())
            if loss_t_t_align is not None:
                msg += ', t_t: {:.5f}'.format(loss_t_t_align.item())
            if loss_v_t_itpl is not None:
                msg += ', v_t_itpl: {:.5f}'.format(loss_v_t_itpl.item())
            # msg += ', time: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(msg)
            if writer != None:
                writer.add_scalar('loss', loss.cpu().detach().numpy(), episode_i)

        # increment in episode_i
        episode_i += cfg.TRAIN.NUM_ROLL_OUT

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

def validate(cfg, model, test_episode_loader, num_test_episodes=300):
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
    ne_list = []  # normalized error

    episode_i = 0
    test_episode_loader.reset()  # clear sampling failure counters

    # if cfg.LOSS.DOMAIN_ALIGNMENT.TYPE != None:
    #     sample_neg_kps_type = cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.TYPE
    #     if sample_neg_kps_type is not None:
    #         negative_point_sampler = PointSampler(type=sample_neg_kps_type,
    #                                     num_kps=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.NUM_PER_IM,
    #                                     dist_thresh=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.DIST_THRESH,
    #                                     bbx_extend_ratio=cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.BBX_EXTEND_RATIO,
    #                                     )
    #     else:
    #         negative_point_sampler = None

    multi_group_supervision = cfg.LOSS.MULTI_GROUP_SUPERVISION  # True or False
    fusing_operation = cfg.LOSS.OBJ_KP_HEATMAP_FUSION  # 'avg' or 'prod'

    while episode_i < num_test_episodes:
        # roll-out an episode
        (supports, support_labels, support_kp_mask, support_scale_trans, _, _, support_saliency, _, _), \
        (queries, query_labels, query_kp_mask, query_scale_trans, _, _, query_saliency, query_bbx_origin, query_w_h_origin), \
        (obj_texts, kps_texts, obj_texts_mask, kps_texts_mask, _, _) = test_episode_loader.next_multi_episodes(s=1)

        # show_save_episode(supports[0], support_labels[0], support_kp_mask[0], queries[0], query_labels[0], query_kp_mask[0], test_episode_loader.episode_generator_chosen, episode_i,
        #                   support_aux_kps=None, support_aux_kp_mask=None, query_aux_kps=None, query_aux_kp_mask=None, is_show=False, is_save=True, delete_old_files=False,
        #                   save_root='output/episode_images', KEYPOINT_TYPES=test_episode_loader.dataset_meta['KEYPOINT_TYPES'])
        #
        # if sample_neg_kps_type is not None:
        #     align_kps, align_kps_mask = get_align_kps_and_mask(support_labels, support_kp_mask, query_labels, query_kp_mask)
        #     sampled_neg_kps = negative_point_sampler(main_kps=align_kps, main_kps_mask=align_kps_mask,
        #                             ims=torch.cat((supports, queries), dim=1), episode_num=episode_i)  # if show imge

        # # --------------------------------------------------------
        # # only used for testing speed when varying number of keypoint
        # S, B1, _ = support_kp_mask.shape
        # _, B2, _ = query_kp_mask.shape
        # N_manual = 1100
        # support_labels = torch.zeros(S, B1, N_manual, 2).cuda()
        # query_labels   = torch.zeros(S, B2, N_manual, 2).cuda()
        # support_kp_mask= torch.ones(S, B1, N_manual).cuda()
        # query_kp_mask  = torch.ones(S, B2, N_manual).cuda()
        # # --------------------------------------------------------

        supports, queries = supports.cuda(), queries.cuda()  # S x B1 x C x H x W, S x B2 x C x H x W
        support_labels, query_labels = support_labels.float().cuda(), query_labels.cuda()  # S x B1 x N x 2, S x B2 x N x 2
        support_kp_mask = support_kp_mask.cuda()  # S x B1 x N
        query_kp_mask = query_kp_mask.cuda()      # S x B2 x N
        obj_texts_mask = obj_texts_mask.cuda()  # S x T1
        kps_texts_mask = kps_texts_mask.cuda()  # S x N x T2

        # # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
        # union_support_kp_mask = torch.sum(support_kp_mask, dim=1) > 0  # S x N
        # # compute the valid query keypoints, using broadcast
        # valid_kp_mask = (query_kp_mask * union_support_kp_mask.unsqueeze(1))  # S x B2 x N
        # num_valid_kps = torch.sum(valid_kp_mask.flatten(1), dim=-1)   # S, valid kps per episode
        # num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=-1)  # S x B2

        outputs = model(supports, queries, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        predict_heatmaps_list = outputs[0]  # [{'obj': tensor, 'text': tensor, 'image': tensor}, ...], a list of dict
        heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect = model.openkd_heatmap_fuse(
            predict_heatmaps_list[0],  # {'obj': tensor, 'text': tensor, 'image': tensor}
            support_kp_mask[0],
            kps_texts_mask[0],
            multi_group_supervision,
            fusing_operation
        )
        heatmaps_predict = heatmaps_fused  # B2 x N x h x w

        # multiple episode images into one (S=1)
        query_labels = query_labels[0]                                # B2 x N x 2
        valid_kp_mask = query_kp_mask[0] * (fused_mask_sum>0).long()  # B2 x N
        query_scale_trans = query_scale_trans[0]                      # B2 x 6
        query_bbx_origin = query_bbx_origin[0]                        # B2 x 4
        query_w_h_origin = query_w_h_origin[0]                        # B2 x 2

        # coordinates decoding
        B2, N = heatmaps_predict.shape[:2]
        if cfg.LOSS.TYPE == 'direct_coord':  # no need to decode
            predictions = heatmaps_predict
        else:
            H, W = heatmaps_predict.shape[2:]
            predict_score, predict_grids = torch.max(heatmaps_predict.reshape(B2, N, -1), 2)  # B2 x N
            predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
            predict_gridxy[:, :, 0] = predict_grids % W  # grid x
            predict_gridxy[:, :, 1] = predict_grids // H  # grid y
            # 'MSE', 'cross-entropy'
            predictions = ((predict_gridxy + 0.5) / H - 0.5) * 2

        predictions = predictions * valid_kp_mask.view(B2, N, 1)
        query_labels = query_labels * valid_kp_mask.view(B2, N, 1)

        predictions = predictions.cpu().detach()
        query_labels = query_labels.cpu().detach()
        valid_kp_mask = valid_kp_mask.cpu().detach()

        # # ----------------------------------------------------------------------------
        # # 1) save predicted keypoints
        # supports = supports.flatten(0, 1).cpu().detach()  # (S * B1) x C x H x W
        # queries = queries.flatten(0, 1).cpu().detach()    # (S * B2) x C x H x W
        # support_labels = support_labels.flatten(0, 1).cpu().detach()  # (S * B1) x N
        # support_kp_mask = support_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        # query_kp_mask = query_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        #
        # save_predictions(supports, support_labels, support_kp_mask, queries, query_labels, valid_kp_mask, predictions,
        #                  test_episode_loader.episode_generator_chosen,episode_i,
        #                  KEYPOINT_TYPES=test_episode_loader.dataset_meta['KEYPOINT_TYPES'],limbs=[])
        #
        # # 2) save multi-group heatmaps / fused heatmaps
        # draw_multi_group_heatmaps = outputs[0]   # a list of G heatmaps, each is (S * B2) x N x h x w
        # draw_multi_group_modulated = outputs[2]  # a list of G heatmaps, each is (S * B2) x N x h x w
        # draw_fused_heatmaps = model.output_fuse(draw_multi_group_heatmaps, fuse_method)  # S x B2 x N x h x w
        # draw_fused_heatmaps = draw_fused_heatmaps.flatten(0, 1).cpu().detach()    # (S * B2) x N x h x w
        # draw_multi_group_heatmaps = list(map(lambda x: x.flatten(0, 1).cpu().detach(), draw_multi_group_heatmaps))
        # draw_multi_group_modulated = list(map(lambda x: x.flatten(0, 1).cpu().detach(), draw_multi_group_modulated))
        # # ``queries" are query images, with size of (S * B2) x C x H x W
        # save_heatmaps(queries, query_labels, valid_kp_mask, predictions, draw_multi_group_modulated, draw_fused_heatmaps,
        #               test_episode_loader.episode_generator_chosen,episode_i)
        # # ----------------------------------------------------------------------------

        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, query_scale_trans)
        query_labels_o = recover_kps(query_labels, square_image_length, query_scale_trans)
        square_diff = torch.sum((predictions_o - query_labels_o) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
        # square_diff2 = torch.sum((predictions / 2 - query_labels /2) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
        if pck_thresh_type == 'bbx':
            longer_edge = np.max(query_bbx_origin[:, [2, 3]].numpy(), axis=1)  # B2, query_bbx_origin's format xmin, ymin, w, h
        else:  # == 'img'
            longer_edge = np.max(query_w_h_origin.numpy(), axis=1)
        longer_edge = longer_edge.reshape(-1, 1)  # B2 x 1

        result_mask = valid_kp_mask.numpy().astype(np.bool_)
        for ind, thr in enumerate(pck_thresh):
            judges = (square_diff <= (thr * longer_edge) ** 2)
            judges = judges.reshape(-1)
            # masking
            judges = judges[result_mask.reshape(-1)]
            tps[ind].extend(judges)
            fps[ind].extend(1 - judges)
            acc_cur = np.sum(judges) / len(judges)
            acc_list[ind].append(acc_cur)

        # compute mean normalized error in each episode
        ne = np.sqrt(square_diff) / np.max(query_w_h_origin.numpy(), axis=1).reshape(-1, 1)
        ne = ne.reshape(-1)
        ne = ne[result_mask.reshape(-1)]
        ne_mean_episode = np.sum(ne) / len(ne)
        ne_list.append(ne_mean_episode)

        if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(tps[0]) > 0:
            # recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
            acc_mean, interval = mean_confidence_interval_multiple(acc_list)
            ne_mean, ne_interval = mean_confidence_interval(ne_list)

            episode_curr = episode_i+1 if (episode_i == (num_test_episodes - 1)) else episode_i
            print('episode {}/{}, Acc {}, Int. {}, NE {:.6f}, Int. {:.6f}, time: {}'.format(episode_curr, num_test_episodes,
                                acc_mean, interval, ne_mean, ne_interval, datetime.datetime.now()))

        # increment in episode_i
        episode_i += 1

    sum_tps = np.sum(np.array(tps), axis=1)
    # recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
    print('episode {}/{}, {}/{}, time: {}'.format(num_test_episodes, num_test_episodes, sum_tps, len(tps[0]), datetime.datetime.now()))
    print('==============testing end================')
    # model.train()
    torch.set_grad_enabled(True)  # enable grad computation

    return acc_mean, interval, ne_mean

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

    output_model_dir, logger, tb_writer = create_loggers(cfg, cfg_str, init_logger=False, init_tb=False)
    # logger.info(cfg)

    print("==>Preparing model")
    openkd_model = get_openkd_model(cfg)
    optimizer = get_optimizer(cfg, openkd_model)
    if cfg.LOSS.TYPE in ['MSE', 'sigmoid-bce', 'cross-entropy']:  # supervised by GT heatmap
        loss_func = HeatmapLoss(cfg)
    elif cfg.LOSS.TYPE == 'direct_coord':  # supervised by GT keypoints
        loss_func = DirectCoordLoss(cfg)
    else:
        raise NotImplementedError

    checkpoint_file = os.path.join(output_model_dir, '%s.pth'%cfg_str)
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        openkd_model.load_state_dict(checkpoint)
        print("==>Loaded checkpoint '{}'".format(checkpoint_file))

    if torch.cuda.is_available():
        openkd_model = openkd_model.cuda()
        loss_func = loss_func.cuda()


    print("==>Preparing data")
    training_kp_category_set, testing_kp_category_set, \
    least_s_kp_num, least_q_kp_num, least_s_kp_num2, least_q_kp_num2 = train_test_kp_set(cfg.DATASET.TYPE)
    # episode_type = cfg.DATASET.EPISODE_TYPE

    # batch_size = cfg.TRAIN.BATCH_SIZE
    episode_type = cfg.DATASET.EPISODE_TYPE  # "one_class", "mix_class"
    k_shot = cfg.TRAIN.NUM_TRAIN_SHOT
    m_query = cfg.TRAIN.NUM_TRAIN_QUERY
    k_shot_test = cfg.TEST.NUM_TEST_SHOT
    m_query_test = cfg.TEST.NUM_TEST_QUERY
    if cfg.DATASET.TYPE in ['ANIMAL_POSE', 'AWA', 'CUB', 'NABIRD']:
        num_train_kp = len(training_kp_category_set)  # used in WG FSL method
        num_test_kp = len(testing_kp_category_set)
        cfg.DATASET[cfg.DATASET.TYPE]['NUM_TRAIN_KP'] = num_train_kp
        cfg.DATASET[cfg.DATASET.TYPE]['NUM_TEST_KP'] = num_test_kp
        n_way = num_train_kp
        n_way_test = num_test_kp

        # seen species, base kps.
        train_episode_loader = build_episode_loader(cfg, training_kp_category_set, n_way, k_shot, m_query,
                            least_s_kp_num, least_q_kp_num, episode_type, phase='train')
        # unseen species, base kps
        val_episode_loader = build_episode_loader(cfg, training_kp_category_set, n_way, k_shot_test, m_query_test,
                            least_s_kp_num, least_q_kp_num, episode_type, phase='val')
        # unseen species, novel kps
        val_episode_loader2 = build_episode_loader(cfg, testing_kp_category_set, n_way_test, k_shot_test, m_query_test,
                                                   least_s_kp_num2, least_q_kp_num2, episode_type, phase='val')
        # unseen species, base kps
        test_episode_loader = build_episode_loader(cfg, training_kp_category_set, n_way, k_shot_test, m_query_test,
                            least_s_kp_num, least_q_kp_num, episode_type, phase='test')
        # unseen species, novel kps
        test_episode_loader2 = build_episode_loader(cfg, testing_kp_category_set, n_way_test, k_shot_test, m_query_test,
                                                   least_s_kp_num2, least_q_kp_num2, episode_type, phase='test')
    elif cfg.DATASET.TYPE in ['DEEPFASHION2']:
        cfg.DATASET[cfg.DATASET.TYPE]['NUM_TRAIN_KP'] = 0
        cfg.DATASET[cfg.DATASET.TYPE]['NUM_TEST_KP'] = 0
        n_way = 0  # number of support keypoints is dynamically determined by sampled clothes category in deepfashion2
        n_way_test = 0

        # seen species, base kps.
        train_episode_loader = build_episode_loader_df(cfg, k_shot, m_query,
                                                       least_s_kp_num, least_q_kp_num, episode_type, phase='train')
        # unseen species, novel kps
        val_episode_loader = build_episode_loader_df(cfg, k_shot, m_query,
                                                     least_s_kp_num, least_q_kp_num, episode_type, phase='val')
        val_episode_loader2 = None
        test_episode_loader = train_episode_loader  # TODO: seen species, base kps
        test_episode_loader2 = val_episode_loader   # TODO: unseen species, novel kps
    else:
        raise NotImplementedError


    num_feature_per_kp = 200
    episode_loader = test_episode_loader
    # episode_loader = train_episode_loader
    kp_labels = test_episode_loader.episode_generator_list[0].support_kp_categories
    kp_num = len(kp_labels)
    root = './output/tSNE'
    if os.path.exists(root) == False:
        os.makedirs(root)
    # No CL
    # filename = 'four2dog_testset_basekps_0shot.pkl'
    # filename = 'four2dog_testset_basekps_1shot.pkl'
    # filename = 'four2dog_testset_basekps_1shot+text.pkl'
    # filename = 'four2dog_testset_basekps_1shot+text_mode2.pkl'
    filename = 'four2dog_testset_basekps_1shot+text+itpl.pkl'
    # filename = 'four2dog_trainset_basekps_1shot+text.pkl'

    # With CL
    # filename = 'four2dog_testset_basekps_1shot+text+itpl_cl.pkl'
    # filename = 'four2dog_testset_basekps_1shot+text+itpl_cl_vt_vv_tt_top1.pkl'
    # filename = 'four2dog_trainset_basekps_1shot+text+itpl_cl.pkl'
    save_path = os.path.join(root, filename)

    # ==================================================================================
    # TODO: grab features via model: N_feat x d, N_feat, N_feat x d, N_feat
    (all_v_features, all_v_inds, v_var_list, all_t_features, all_t_inds, t_var_list) = \
        grab_features(cfg, openkd_model, episode_loader, num_feature_per_kp, mode=1)
    print(v_var_list)
    print(t_var_list)

    feat_dict = {'v_feat': all_v_features, 'v_ind': all_v_inds, 'v_var': v_var_list,
                 't_feat': all_t_features, 't_ind': all_t_inds, 't_var': t_var_list}
    pickle.dump(feat_dict, open(save_path, 'wb'))

    # ==================================================================================
    # TODO: plot average variance curves
    feat_dict = pickle.load(open(save_path, 'rb'))
    v_var_list, t_var_list = feat_dict['v_var'], feat_dict['t_var']

    # t_var_list = pickle.load(open(os.path.join(root, 'four2dog_testset_basekps_0shot.pkl'), 'rb'))['t_var']
    # v_var_list = pickle.load(open(os.path.join(root, 'four2dog_testset_basekps_1shot.pkl'), 'rb'))['v_var']

    plt.rcParams['font.size'] = 14
    config = {
        "mathtext.fontset": 'cm',  # 'cm' (default font in latex) or 'stix' (similar to Times New Roman)
    }
    plt.rcParams.update(config)
    fig, ax = plt.subplots()

    ax.plot(np.arange(len(v_var_list)), v_var_list, linewidth=3, label='var_visual')
    ax.plot(np.arange(len(t_var_list)), t_var_list, linewidth=3, label='var_textual')

    # set ticks, ticklabels and their position
    ax.set_xticks(np.arange(len(kp_labels)))
    ax.set_xticklabels(kp_labels, fontsize=14)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    # plt.ylim(0, 2.0)
    # ax.set_xlabel(r'Keypoint types', fontsize=14)
    ax.set_ylabel(r'Feature variance', fontsize=18)
    # plt.title('Variance per keypoint for visual & text modality') #, fontsize=14)
    ax.legend(fontsize=14)
    plt.grid(axis='both', linestyle='--', c='grey', alpha=0.4)

    save_im_path = os.path.join(root, filename.split('.')[0] + '_var.pdf')
    plt.savefig(save_im_path, bbox_inches='tight')
    plt.show()

    # ==================================================================================
    # TODO: load previously saved data
    feat_dict = pickle.load(open(save_path, 'rb'))
    (all_v_features, all_v_inds, v_var_list, all_t_features, all_t_inds, t_var_list) = \
        (feat_dict['v_feat'], feat_dict['v_ind'], feat_dict['v_var'], \
         feat_dict['t_feat'], feat_dict['t_ind'], feat_dict['t_var'])

    # feat_dict0 = pickle.load(open(os.path.join(root, 'four2dog_testset_basekps_0shot.pkl'), 'rb'))
    # (all_t_features, all_t_inds) = (feat_dict0['t_feat'], feat_dict0['t_ind'])
    # feat_dict1 = pickle.load(open(os.path.join(root, 'four2dog_testset_basekps_1shot.pkl'), 'rb'))
    # (all_v_features, all_v_inds) = (feat_dict1['v_feat'], feat_dict1['v_ind'])

    # ==================================================================================
    # TODO: plot distribution via tSNE
    tSNE_v = TSNE(n_components=2, init='pca', random_state=0, perplexity=40, early_exaggeration=12, verbose=True)
    tSNE_t = TSNE(n_components=2, init='pca', random_state=0, perplexity=num_feature_per_kp, early_exaggeration=12, verbose=True)

    V = tSNE_v.fit_transform(all_v_features)  # N_feat x 2
    V2 = np.concatenate((V, all_v_inds.reshape(-1, 1)), axis=1)  # N_feat

    T = tSNE_t.fit_transform(all_t_features)  # N_feat x 2
    T2 = np.concatenate((T, all_t_inds.reshape(-1, 1)), axis=1)  # N_feat

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    h1 = ax[0].scatter(V2[:, 0], V2[:, 1], c=V2[:, 2].astype(np.int16),
                marker='.',
                cmap=plt.cm.get_cmap('tab20', kp_num)
                # 'cubehelix', 'tab20', 'tab10'. cmap_tmp = plt.cm.get_cmap('tab20'), then use color as cmap_tmp(i)
                )
    h2 = ax[1].scatter(T2[:, 0], T2[:, 1], c=T2[:, 2].astype(np.int16),
                marker='.',
                cmap=plt.cm.get_cmap('tab20', kp_num),
                clim=(-0.5, kp_num - 0.5),
                # 'cubehelix', 'tab20', 'tab10'. cmap_tmp = plt.cm.get_cmap('tab20'), then use color as cmap_tmp(i)
                )
    # cbar = plt.colorbar(ticks=range(11), label='body part')
    # cbar = plt.colorbar()
    cbar = plt.colorbar(h2, ax=ax, orientation='vertical')  #, shrink=1.0)
    # plt.clim(-0.5, kp_num - 0.5)  # set color limits in [-0.5, 10.5]
    ticks = np.arange(kp_num)
    cbar.set_ticks(ticks=ticks, labels=kp_labels)  # set ticks and tick labels
    # cbar.set_label('Colorbar label')

    ax[0].set_xlabel('Visual prompting (1-shot)',  fontsize=14)
    ax[1].set_xlabel('Textual prompting (0-shot)', fontsize=14)
    save_im_path = os.path.join(root, filename.split('.')[0] + '.pdf')
    plt.savefig(save_im_path, bbox_inches='tight')
    plt.show()

def grab_features(cfg, model, test_episode_loader, num_feature_per_kp=300, mode=1):
    '''
    Extract visual kp and text features in two modes:
    mode 1: ensure every keypoint type has same number of valid features. we will feed as many as possible episodes.
    mode 2: simply draw a number of episodes (note the features per keypoint-type may not be balanced.)
    '''
    print('==============testing start==============')
    torch.set_grad_enabled(False)  # disable grad computation
    # model.eval()

    features_list = []
    kp_inds_list = []

    episode_i = 0
    test_episode_loader.reset()  # clear sampling failure counters

    multi_group_supervision = cfg.LOSS.MULTI_GROUP_SUPERVISION  # True or False
    fusing_operation = cfg.LOSS.OBJ_KP_HEATMAP_FUSION  # 'avg' or 'prod'

    n_way = test_episode_loader.n_way
    kp_categories = test_episode_loader.episode_generator_list[0].support_kp_categories
    v_repres_dict = {}
    t_repres_dict = {}
    for n in range(n_way):
        v_repres_dict[n] = []
        t_repres_dict[n] = []
    while True:
        # roll-out an episode
        (supports, support_labels, support_kp_mask, support_scale_trans, _, _, support_saliency, _, _), \
        (queries, query_labels, query_kp_mask, query_scale_trans, _, _, query_saliency, query_bbx_origin, query_w_h_origin), \
        (obj_texts, kps_texts, obj_texts_mask, kps_texts_mask, _, _) = test_episode_loader.next_multi_episodes(s=1)

        supports, queries = supports.cuda(), queries.cuda()  # S x B1 x C x H x W, S x B2 x C x H x W
        support_labels, query_labels = support_labels.float().cuda(), query_labels.cuda()  # S x B1 x N x 2, S x B2 x N x 2
        support_kp_mask = support_kp_mask.cuda()  # S x B1 x N
        query_kp_mask = query_kp_mask.cuda()      # S x B2 x N
        obj_texts_mask = obj_texts_mask.cuda()  # S x T1
        kps_texts_mask = kps_texts_mask.cuda()  # S x N x T2
        
        v_repres_list, t_repres_list = inference(model, supports, queries, support_labels, support_kp_mask, obj_texts, obj_texts_mask, kps_texts, kps_texts_mask)
        assert len(v_repres_list) > 0, 'visual prompts are required.'
        assert len(t_repres_list) > 0, 'textual prompts are required.'

        if mode == 1:
            #========================================================
            # Mode 1: filter invalid visual kp repres or textual repres

            v_flag, t_flag = True, True  # judge if gathered features

            if len(v_repres_list) > 0:
                v_repres = v_repres_list[0]  # B1 x N x d
                v_mask = support_kp_mask[0]  # B1 x N
                B1, N = v_mask.shape
                for n in range(N):
                    if len(v_repres_dict[n]) >= num_feature_per_kp:
                        continue
                    v_flag = False
                    for b in range(B1):
                        if v_mask[b, n] > 0:
                            v_repres_dict[n].append(v_repres[b, :, n])
                        if len(v_repres_dict[n]) >= num_feature_per_kp:
                            break

            if len(t_repres_list) > 0:
                t_repres = t_repres_list[0]  # (N*T2) x d
                t_mask = kps_texts_mask[0]   # N x T2
                N, T2 = t_mask.shape
                for n in range(N):
                    if len(t_repres_dict[n]) >= num_feature_per_kp:
                        continue
                    t_flag = False
                    for t in range(T2):
                        if t_mask[n, t] > 0:
                            t_repres_dict[n].append(t_repres[n*T2+t])
                        if len(t_repres_dict[n]) >= num_feature_per_kp:
                            break

            if v_flag == True and t_flag == True:  # finish gathering features, break ``while"
                break
            # ========================================================

        elif mode == 2:
            # ========================================================
            # Mode 2
            if episode_i >= num_feature_per_kp:
                break
            v_repres = v_repres_list[0]  # B1 x N x d
            v_mask = support_kp_mask[0]  # B1 x N
            B1, N = v_mask.shape
            for n in range(N):
                for b in range(B1):
                    if v_mask[b, n] > 0:
                        v_repres_dict[n].append(v_repres[b, :, n])

            t_repres = t_repres_list[0]  # (N*T2) x d
            t_mask = kps_texts_mask[0]   # N x T2
            N, T2 = t_mask.shape
            for n in range(N):
                for t in range(T2):
                    if t_mask[n, t] > 0:
                        t_repres_dict[n].append(t_repres[n * T2 + t])
            # ========================================================
        else:
            raise NotImplementedError

        if (episode_i) % 20 == 0:
            print('episode: ', episode_i)

        # increment in episode_i
        episode_i += 1

    all_v_features = []
    all_v_inds = []
    all_t_features = []
    all_t_inds = []
    v_var_list = []
    t_var_list = []
    for n in v_repres_dict.keys():
        v_repres_tmp = torch.stack(v_repres_dict[n], dim=0)  # num_feature_per_kp x d
        num_tmp = len(v_repres_dict[n])
        v_inds_tmp = torch.zeros(num_tmp).fill_(n)  # num_feature_per_kp
        all_v_features.append(v_repres_tmp)
        all_v_inds.append(v_inds_tmp)

        v_var = compute_var(v_repres_tmp)
        v_var_list.append(v_var)

    all_v_features = torch.cat(all_v_features).cpu().detach().numpy()
    all_v_inds = torch.cat(all_v_inds).cpu().detach().numpy()

    for n in t_repres_dict.keys():
        t_repres_tmp = torch.stack(t_repres_dict[n], dim=0)
        num_tmp = len(t_repres_dict[n])
        t_inds_tmp = torch.zeros(num_tmp).fill_(n)
        all_t_features.append(t_repres_tmp)
        all_t_inds.append(t_inds_tmp)

        t_var = compute_var(t_repres_tmp)
        t_var_list.append(t_var)

    all_t_features = torch.cat(all_t_features).cpu().detach().numpy()
    all_t_inds = torch.cat(all_t_inds).cpu().detach().numpy()

    print('==============testing end================')
    # model.train()
    torch.set_grad_enabled(True)  # enable grad computation

    return (all_v_features, all_v_inds, v_var_list, all_t_features, all_t_inds, t_var_list)

def compute_var(features):
    # features: N x d
    m = features.mean(0)  # d
    var = ((features - m.reshape(1, -1)) ** 2).mean(0).mean(0)
    var = var.cpu().detach().numpy()
    return var

def inference(model, supports_, queries_, support_kps_=None, support_kp_mask_=None, obj_texts_=((),), obj_texts_mask_=None,
                kps_texts_=((),), kps_texts_mask_=None, itpl_texts_pool_=((),), itpl_texts_pool_mask_=None, **kwargs):
    B1 = 0 if (supports_ is None) or (len(supports_[0].shape) !=4) else supports_[0].shape[0]  # Judge whether it is zero-shot or not
    S, B2, C, H, W = queries_.shape   # S episodes, B2 query images
    B_total = B1 + B2

    if B1 > 0:  # has visual prompt
        N_v = support_kp_mask_.shape[-1] # N_v visual prompted keypoints
    else:
        N_v = 0

    T1 = len(obj_texts_[0])           # T1, number of texts per object
    if len(kps_texts_[0]) > 0:
        assert kps_texts_mask_ is not None
        _, N_t, T2 = kps_texts_mask_.shape  # T2, number of texts per kp
    else:
        N_t, T2 = 0, 0

    if len(itpl_texts_pool_[0]) > 0:
        assert itpl_texts_pool_mask_ is not None
        _, N_path, T3 = itpl_texts_pool_mask_.shape  # T3, number of texts per path
    else:
        N_path, T3 = None, 0

    assert N_v > 0 or N_t > 0, 'The number of visual prompts or textual prompts should > 0.'

    # combine images
    if B1 != 0:  # Has visual prompt
        in_ims = torch.cat([supports_, queries_], dim=1)  # S x (B1+B2) x C x H x W
        in_ims = in_ims.reshape(S * (B1 + B2), C, H, W)  # (S * (B1+B2)) x C x H x W
    else:  # Do not have visual prompt (zero-shot)
        in_ims = queries_.reshape(S * B2, C, H, W)  # (S * B2) x C x H x W

    # combine texts
    in_obj_texts = []  # (S*T1) texts
    in_kps_texts = []  # (S*N_t*T2) texts
    in_itpl_texts = [] # (S*N_path*T3) texts
    for s in range(S):
        if T1 > 0:
            in_obj_texts += obj_texts_[s]
        if T2 > 0:
            in_kps_texts += kps_texts_[s]
        if T3 > 0:
            in_itpl_texts += itpl_texts_pool_[s]
    in_texts = in_obj_texts + in_kps_texts + in_itpl_texts  # {S*T1 + S*(N_t*T2) + S*(N_path*T3)} texts

    # TODO: 1) Image and text features extraction
    if model.trunk == 'CLIP':
        # TODO: test CLIP's matching ability between images and texts
        # text_collection = ["a diagram", "a dog", "left ear", "triangles", "the horse", "cat", "cow", "sheep"]
        # text = clip.tokenize(text_collection).to('cuda')
        # logits_per_image, logits_per_text = model.encoder(queries_[0].cuda(), text)
        # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        # print("Label probs:", probs)

        # 1) clip's text tokenize: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 77
        in_texts_tokens = clip.tokenize(in_texts).cuda() if len(in_texts) > 0 else []

        # 2) extract text features: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 77 x d (CLIP)
        if len(in_texts_tokens) > 0:
            out_texts_features = model.encoder.encode_text(in_texts_tokens)
        else:
            out_texts_features = None
        # extract image features (after_proj and before_proj features)
        # Support+query: (S * (B1+B2)) x (1+H*W) x C or Query image: (S * B2) x (1+H*W) x C
        out_ims_features = model.encoder.encode_image(in_ims)

        # 3) freeze image and text features
        # out_ims_features = list(map(lambda x: x.detach(), out_ims_features))
        # if out_texts_features is not None:
        #     out_texts_features = list(map(lambda x: x.detach(), out_texts_features))

    elif model.trunk == 'BLIP':
        # TODO: test BLIP's matching ability between images and texts
        # text_collection = ["a diagram", "a dog", "left ear", "triangles", "the horse", "cat", "cow", "sheep"]
        # # image-text contrastive learning's similarity
        # sim = model.encoder(queries_[0].cuda(), text_collection, match_head='itc')
        # probs = torch.nn.functional.softmax(sim / model.encoder.temp, dim=1).detach().cpu().numpy()  # 5 x 8
        # print('The image and text\'s matching probs: \n', probs)
        # # image-text matching via a binary classifier
        # itm = model.encoder(queries_[0].cuda(), text_collection[-5:], match_head='itm')
        # itm_score = torch.nn.functional.softmax(itm, dim=1)[:, 1]  # 5 x 2 --> 5 x 1
        # itm_score = itm_score.detach().cpu().numpy()
        # print('The image and text\'s matching score: \n', itm_score)

        # 1) blip's text tokenize: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 35
        (in_texts_tokens, in_texts_tokens_mask) = model.encoder.tokenize_batch_texts(in_texts, device='cuda') if len(in_texts) > 0 else ([], [])

        # 2) extract text features: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 35 x d (BLIP)
        if len(in_texts_tokens) > 0:
            out_texts_features = model.encoder.encode_batch_texts(in_texts_tokens, in_texts_tokens_mask)
        else:
            out_texts_features = None
        # extract image features (after_proj and before_proj features)
        # Support+query: (S * (B1+B2)) x (1+H*W) x C or Query image: (S * B2) x (1+H*W) x C
        out_ims_features = model.encoder.encode_image(in_ims)

        # 3) freeze image and text features
        # out_ims_features = list(map(lambda x: x.detach(), out_ims_features))
        # if out_texts_features is not None:
        #     out_texts_features = list(map(lambda x: x.detach(), out_texts_features))
    else:
        raise NotImplementedError

    # TODO: upscale image feature map
    if model.feature_sr is not None:
        out_ims_features_0 = model.feature_map_upscale(out_ims_features[0])  # (S*B) x (1+r*H*r*W) x C
        out_ims_features = [out_ims_features_0]

    # TODO: 2) Feature adaptation
    feat_width = int(np.sqrt(out_ims_features[0].shape[1] - 1))
    assert (feat_width ** 2 + 1) == out_ims_features[0].shape[1], 'Feature resolution is not right.'
    if model.adaptation_net_type == None:
        adapted_ims_features = out_ims_features[0]
        adapted_texts_features = out_texts_features[0] if out_texts_features is not None else None
    elif model.adaptation_net_type == 'RESIDUAL_REFINE':  # (use after_proj and before_proj features)
        adapted_ims_features = model.vision_anet(out_ims_features[0], out_ims_features[1], h=feat_width, w=feat_width)
        adapted_texts_features = model.text_anet(out_texts_features[0], out_texts_features[1]) if out_texts_features is not None else None
    elif model.adaptation_net_type == 'RESIDUAL_REFINE2':  # (use after_proj features)
        adapted_ims_features = model.vision_anet(out_ims_features[0], out_ims_features[0], h=feat_width, w=feat_width)
        adapted_texts_features = model.text_anet(out_texts_features[0], out_texts_features[0]) if out_texts_features is not None else None
    else:
        raise NotImplementedError

    enable_alignment = (model.alignment_type is not None) and (kwargs.get('num_main_kps') is not None) and (kwargs.get('query_kps_') is not None)
    if enable_alignment == True:
        support_repres_list = []
        query_repres_list = []
        kps_text_proto_list = []
        kps_text_proto_mask_list = []

    visual_repres_list = []
    textual_repres_list = []
    model.set_complement_itpl_kps_texts_info()
    for epi_ind in range(S):
        # TODO: 3) Keypoint prompt set building
        prompt_set = {'obj': [], 'text': [], 'image': []}
        # Parse encoded text features. We may have text features or not, depending on given prompts.
        # adapted_texts_features has size of {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x L x d
        if len(in_obj_texts) > 0:
            start_ind = epi_ind*T1
            end_ind   = (epi_ind+1)*T1
            if model.trunk == 'CLIP':  # for clip, text CLS token is at last
                obj_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # T1 x D
            elif model.trunk == 'BLIP':  # for blip, text CLS token is at first
                obj_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind)][:, 0, :]  # T1 x D
            # obj_text_proto = obj_text_CLS.mean(dim=0, keepdim=True)  # 1 x D
            obj_text_mask_per_episode = obj_texts_mask_[epi_ind]  # T1
            obj_text_mask_per_episode_sum = obj_text_mask_per_episode.sum()  # a scalar
            obj_text_proto = (obj_text_CLS * obj_text_mask_per_episode.unsqueeze(-1)).sum(dim=0, keepdim=True) # 1 x D
            if obj_text_mask_per_episode_sum > 0:
                obj_text_proto /= obj_text_mask_per_episode_sum  # 1 x D
            prompt_set['obj'] = obj_text_proto  # 1 x D
        else:
            obj_text_proto = []
        if len(in_kps_texts) > 0:
            base_ind = len(in_obj_texts)
            start_ind = base_ind + epi_ind*N_t*T2
            end_ind   = base_ind + (epi_ind+1)*N_t*T2
            if model.trunk == 'CLIP':  # for clip, text CLS token is at last
                kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # (N_t*T2) x D
            elif model.trunk == 'BLIP':  # for blip, text CLS token is at first
                kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind)][:, 0, :]  # (N_t*T2) x D

            textual_repres_list.append(kps_text_CLS)

            # kps_text_mask_per_episode = kps_texts_mask_[epi_ind]  # N_t x T2
            # # kps_text_proto = kps_text_CLS.reshape(N, T2, -1).mean(dim=1)  # N_t * D
            # kps_text_mask_per_episode = kps_texts_mask_[epi_ind]  # N_t x T2
            # kps_text_mask_per_episode_sum = kps_text_mask_per_episode.sum(dim=-1)  # N_t
            # kps_text_proto = (kps_text_CLS.reshape(N_t, T2, -1) * kps_text_mask_per_episode.unsqueeze(-1)).sum(dim=1)  # N_t * D
            # kps_text_mask_per_episode_tmp = (kps_text_mask_per_episode_sum <= 0).long() + kps_text_mask_per_episode_sum  # +1 to avoid dividing zero
            # kps_text_proto /= kps_text_mask_per_episode_tmp.unsqueeze(-1)  # N_t * D
            # kps_text_proto_mask = (kps_text_mask_per_episode_sum > 0).long()  # N_t
            # prompt_set['text'] = kps_text_proto  # N_t * D
        else:
            kps_text_proto = []

        # Parse encoded image features. We definitely have query features, but may have support image features or not
        # Support+query: (S * (B1+B2)) x (1+H*W) x C or Query image: (S * B2) x (1+H*W) x C
        if B1 > 0:
            # support image (visual prompt): B1 x (1+h*w) x C
            support_im_tokens = adapted_ims_features[epi_ind*B_total: epi_ind*B_total+B1]
            support_kps = support_kps_[epi_ind]          # B1 x N_v x 2, ranges -1~1 (continuous)
            support_kp_mask = support_kp_mask_[epi_ind]  # B1 x N_v

            # query image: B2 x (1+h*w) x C
            query_im_tokens   = adapted_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]

            support_features = (support_im_tokens[:, 1:]).permute(0, 2, 1).reshape(B1, -1, feat_width, feat_width)  # B1 x C x h x w, remove CLS
            # TODO: Note we may bring in human bias when extracting keypoint representations here
            support_repres = model.visual_prompt_extraction(model.visual_prompt_extraction_type, support_features, support_kps, support_kp_mask, W)  # B1 x C x N_v

            visual_repres_list.append(support_repres)

            # avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N_v
            # kps_visual_proto = avg_support_repres.transpose(1, 0)  # N_v x C
            # prompt_set['image'] = kps_visual_proto  # TODO: Note not all visual proto are valid. Need masking when fusing

            # if enable_alignment == True:  # hook features for domain alignment
            #     support_repres_list.append(support_repres)  # B1 x C x N_v
        else:
            support_im_tokens = []
            kps_visual_proto = []
            # query image: B2 x (1+h*w) x C
            query_im_tokens = adapted_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]
        query_features = (query_im_tokens[:, 1:]).permute(0, 2, 1).reshape(B2, -1, feat_width, feat_width)  # B2 x C x h x w, remove CLS

    return visual_repres_list, textual_repres_list


if __name__ == '__main__':
    main()