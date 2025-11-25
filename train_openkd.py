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
from datasets.build_dataset import build_episode_loader, build_dataset_meta
from datasets.dataset_utils import draw_instance, draw_skeletons, draw_markers
from datasets.gt_itpl_texts import groundtruth_itpl_texts
from network.openkd_model import get_openkd_model, OpenKDModel
import network.clip_kd as clip
from core.loss_lw import HeatmapLoss, DirectCoordLoss
from core.misc import compute_openkd_heatmap_loss, split_main_aux_heatmaps
from vis import show_save_episode, save_predictions, save_heatmaps

import cv2

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
            eval_results = validate(cfg, model, test_episode_loader, num_test_episodes=300)
            recall = eval_results[0]
            ne = eval_results[2]
            if test_episode_loader2 != None:
                eval_results2 = validate(cfg, model, test_episode_loader2, num_test_episodes=300)
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

        #=====================================================
        # TODO: Used to test upper bound for GT interpolated texts
        # itpl_texts_pool, _ = groundtruth_itpl_texts(dataset_type=cfg.DATASET.TYPE)
        # itpl_texts_pool = [copy.deepcopy(itpl_texts_pool) for _ in range(cfg.TRAIN.NUM_ROLL_OUT)]
        # itpl_texts_pool_mask = itpl_texts_pool_mask[:, :, 0:1]  # set T3=1
        # =====================================================


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
                            global_episode_cnt=episode_i,       # used for picking interpolated text
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

def validate(cfg, model, test_episode_loader, num_test_episodes=300, **kwargs):
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
        # B1 = 0 if (supports is None) or (len(supports[0].shape) != 4) else supports[0].shape[0]  # Judge whether it is zero-shot or not
        # supports = None if B1<=0 else supports.flatten(0, 1).cpu().detach()  # (S * B1) x C x H x W
        # support_labels = None if B1<=0 else support_labels.flatten(0, 1).cpu().detach()  # (S * B1) x N
        # support_kp_mask = None if B1<=0 else support_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        # kps_texts = kps_texts[0]  # a list of an episode's texts (N_t*T2) or []
        # queries = queries.flatten(0, 1).cpu().detach()  # (S * B2) x C x H x W
        # query_kp_mask = query_kp_mask.flatten(0, 1).cpu().detach()  # (S * B1) x N
        #
        # save_predictions(supports, support_labels, support_kp_mask, queries, query_labels, valid_kp_mask, predictions,
        #                  test_episode_loader.episode_generator_chosen,episode_i,
        #                  KEYPOINT_TYPES=test_episode_loader.dataset_meta['KEYPOINT_TYPES'],limbs=[], kp_texts=kps_texts)
        #
        # # 2) save multi-group heatmaps / fused heatmaps
        # predict_heatmaps_dict = predict_heatmaps_list[0]  # a dict: {'obj': [], 'text': [], 'image': []}
        # draw_multi_group_heatmaps = []  # model.dict2list(predict_heatmaps_dict)  # a list of G heatmaps, each is B2 x N x h x w
        # draw_fused_heatmaps = heatmaps_fused.cpu().detach()    # B2 x N x h x w
        # draw_multi_group_heatmaps = list(map(lambda x: x.cpu().detach(), draw_multi_group_heatmaps))
        #
        # # ``queries" are query images, with size of B2 x C x H x W
        # save_heatmaps(queries, query_labels, valid_kp_mask, predictions, draw_multi_group_heatmaps, draw_fused_heatmaps,
        #               test_episode_loader.episode_generator_chosen,episode_i, kp_texts=kps_texts)
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

    output_model_dir, logger, tb_writer = create_loggers(cfg, cfg_str)
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

    print("==> OpenKD model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in openkd_model.parameters() if p.requires_grad]):,}")

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



    # train epochs
    meter = {
        'recall_stack': [0, 0],
        'recall_best': 0,
        'ne_stack': [1, 1],
        'ne_best': 1,
    }
    train_episodes(cfg, openkd_model, loss_func, optimizer,
                   train_episode_loader, val_episode_loader, test_episode_loader2=val_episode_loader2,
                   writer=tb_writer, checkpoint_file=checkpoint_file, meter=meter)

    eval_cost = True
    openkd_model.set_cost_eval(eval_cost)

    print('==>Training finished!')
    print('==>Final testing, loading best model')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        openkd_model.load_state_dict(checkpoint)
        print("==>Loaded checkpoint '{}'".format(checkpoint_file))
    print('==>Test2-unseen species, base kps')
    eval_results = validate(cfg, openkd_model, test_episode_loader, num_test_episodes=1000)
    acc = eval_results[0]

    cost_base = openkd_model.get_cost_eval()  # record results
    print('cost_base:', cost_base)
    avg_it_base = (cost_base['IT1'] + cost_base['IT2'] + cost_base['IT3'])
    print('Avg IT:  %.6f sec/episode'%(avg_it_base))
    openkd_model.set_cost_eval(eval_cost)

    print('==>Test3-unseen species, novel kps')
    eval_results = validate(cfg, openkd_model, test_episode_loader2, num_test_episodes=1000)
    acc = eval_results[0]

    cost_novel = openkd_model.get_cost_eval()  # record results
    print('cost_novel:', cost_novel)
    avg_it_novel = (cost_novel['IT1'] + cost_novel['IT2'] + cost_novel['IT3'])
    print('Avg IT:  %.6f sec/episode' % (avg_it_novel))
    print('Final Avg IT: %.6f sec/episode | IT1: %.6f, IT2: %.6f, IT3: %.6f'%((avg_it_novel+avg_it_base)/2, \
          (cost_base['IT1']+cost_novel['IT1'])/2, (cost_base['IT2']+cost_novel['IT2'])/2, (cost_base['IT3']+cost_novel['IT3'])/2))

def get_dataset_type(cfg):
    if cfg.DATASET.TYPE == 'ANIMAL_POSE':
        return cfg.DATASET.ANIMAL_POSE.UNSEEN_CLASS
    else:
        return str(cfg.DATASET.TYPE).lower()

if __name__ == '__main__':
    main()