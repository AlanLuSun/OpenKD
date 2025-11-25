# -*- coding: utf-8 -*-
import copy
import os
import torch
from torch import nn
from torchvision.models.resnet import resnet34, resnet50
import numpy as np
from network.models_gridms2 import extract_representations, average_representations2
import logging
from utils.utils import SoftArgmax, compute_similarity, compute_similarity2, compute_distance, AverageMeter, apply_noise
import time

import network.dino_vit as  DINO_VIT

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, trunk='resnet50', layer_to_freezing=-1, downsize_factor=16, specify_output_channel=None):
        super(Encoder, self).__init__()

        if trunk == 'resnet50':
            # Bottleneck numbers [3, 4, 6, 3] for layer 1, 2, 3, 4
            resnet = resnet50(pretrained=True)

            if downsize_factor == 16:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3._modules['0'],
                    resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                    # resnet.layer3,
                    # resnet.layer4._modules['0'],
                    # resnet.layer4._modules['1'],    # feature size B x 2048 x 12 x 12, 1/32 (368 x 368)
                )
                feature_dim = 1024  # B x 1024 x 23 x 23 (1/16)
            elif downsize_factor == 32:
                self.encoder_list = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    # resnet.layer3._modules['0'],
                    # resnet.layer3._modules['1'],  # feature size B x 1024 x 23 x 23, 1/16 (368 x 368)
                    resnet.layer3,
                    resnet.layer4._modules['0'],
                    resnet.layer4._modules['1'],    # feature size B x 2048 x 12 x 12, 1/32 (368 x 368)
                    # resnet.layer4
                )
                feature_dim = 2048  # B x 2048 x 12 x 12 (1/32)

            self.feature_dim = feature_dim
            self.freeze = layer_to_freezing

            self.specify_output_channel = specify_output_channel
            if specify_output_channel != None:
                self.end_conv1x1 = nn.Sequential(
                    nn.Conv2d(feature_dim, specify_output_channel, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(specify_output_channel),
                    nn.ReLU(True)
                )

            # ================================
            # manually free up used GPU valirable and clean catche added by Changsheng Lu, 2023.2.16
            # del resnet
            # torch.cuda.empty_cache()
            # ================================
        print('Encoder initialized!')
        # print(self.encoder_list)

    def forward(self, x: torch.Tensor):
        """feedforward propagation

        Arguments:
            x {torch.Tensor} -- B × 3 × H × W

        Returns:
            x [torch.Tensor] -- B × 3 × h × w
        """

        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)
            if self.freeze == i:
                x = x.detach()

        if self.specify_output_channel != None:
            x = self.end_conv1x1(x)

        return x

class EncoderMultiScale(nn.Module):
    def __init__(self, trunk='RESNET50', layer_to_freezing=-1, downsize_factor_list=[8, 16, 32], **kwargs):
        super(EncoderMultiScale, self).__init__()

        if trunk == 'RESNET50':
            # Bottleneck numbers [3, 4, 6, 3] for layer 1, 2, 3, 4
            resnet = resnet50(pretrained=True)

            self.encoder_list = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # L4
                resnet.layer2._modules['0'],  # L5, feature size B x 512 x 48 x 48, 1/8 (384 x 384)
                resnet.layer2._modules['1'],  # L6
                resnet.layer2._modules['2'],  # L7
                resnet.layer2._modules['3'],  # L8
                resnet.layer3._modules['0'],  # L9, feature size B x 1024 x 24 x 24, 1/16 (384 x 384)
                resnet.layer3._modules['1'],  # L10
                resnet.layer3._modules['2'],
                resnet.layer3._modules['3'],
                resnet.layer3._modules['4'],
                resnet.layer3._modules['5'],  # L14
                resnet.layer4._modules['0'],  # L15, feature size B x 2048 x 12 x 12, 1/32 (384 x 384)
                resnet.layer4._modules['1'],  # L16
            )
            ds_dim_dict = {8: 512, 16: 1024, 32: 2048}  # downsize_factor: feature_dim
            ds_layer_dict = dict(zip([8, 16, 32], kwargs['layer_to_grab_features']))  # downsize_factor: layer_index
            self.grab_index = []
            self.feature_dims = []
            downsize_factor_list.sort()
            for ds in downsize_factor_list:
                self.grab_index.append(ds_layer_dict[ds])
                self.feature_dims.append(ds_dim_dict[ds])
            if len(self.grab_index) == 0:
                raise ValueError
            self.freeze = layer_to_freezing

            # self.specify_output_channel = specify_output_channel
            # if specify_output_channel != None:
            #     self.end_conv1x1 = nn.Sequential(
            #         nn.Conv2d(feature_dim, specify_output_channel, kernel_size=1, stride=1, padding=0),
            #         nn.BatchNorm2d(specify_output_channel),
            #         nn.ReLU(True)
            #     )

            # ================================
            # manually free up used GPU valirable and clean catche added by Changsheng Lu, 2023.2.16
            # del resnet
            # torch.cuda.empty_cache()
            # ================================
        print('Encoder initialized!')
        # print(self.encoder_list)

    def forward(self, x: torch.Tensor):
        """feedforward propagation

        Arguments:
            x {torch.Tensor} -- B × 3 × H × W

        Returns:
            x [torch.Tensor] -- B × 3 × h × w
        """
        features = []
        ind = 0
        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)
            if self.freeze == i:
                x = x.detach()
            if self.grab_index[ind] == i:
                features.append(x)
                if len(features) < len(self.grab_index):
                    ind += 1

        # if self.specify_output_channel != None:
        #     x = self.end_conv1x1(x)

        return features

# base localization network whose weights are fixed
class LocalizationNetBase(nn.Module):
    def __init__(self, num_base, feature_channel=2048, normalize_weights=False):
        super(LocalizationNetBase, self).__init__()
        self.num_base = num_base
        self.conv = nn.Conv2d(feature_channel, num_base, kernel_size=1, stride=1, padding=0, bias=False)  # no biase I set
        self.conv.weight.data.normal_(0, np.sqrt(2.0 / feature_channel))

        # weather normalize weights during feedforward inference
        self.normalize_weights = normalize_weights

    def forward(self, feature: torch.Tensor):
        '''
        feature: B x C x H x W
        '''
        if self.normalize_weights == False:
            out = self.conv(feature)
        else:
            N = self.num_base
            B, C, H, W = feature.shape

            weights = self.conv.weight #.data
            assert weights.shape[0] == N and weights.shape[1] == C, "Feature dim is not right. Error in LocalizationNetBase."
            weights = weights.reshape(N, C)
            weights = torch.nn.functional.normalize(weights, p=2, dim=1, eps=1e-12)  # N x C

            feature_t = torch.nn.functional.normalize(feature.view(B, C, H*W), p=2, dim=1, eps=1e-12)  # B x C x (h*w)

            out = torch.bmm(weights.unsqueeze(0).expand(B, N, C), feature_t)
            out = out.view(B, N, H, W)
            # out = (out + 1) / 2.0  # range -1~1 --> 0~1

        return out

    def get_conv_weights(self):
        N = self.num_base
        weights = self.conv.weight
        weights = weights.reshape(N, -1)
        # if self.normalize_weights == False:
        #     pass
        # else:
        #     weights = torch.nn.functional.normalize(weights, p=2, dim=1, eps=1e-12)  # N x C
        return weights

# dynamic localization network whose weights are predicted from weight generator
class LocalizationNet(nn.Module):
    def __init__(self, normalize_weights = False):
        super(LocalizationNet, self).__init__()
        self.normalize_weights = normalize_weights

        # self.conv = nn.Conv2d(2048, 11, 1, 1, 0)
        # self.conv.weight.data.normal_(0, 0.01)

    def forward(self, feature: torch.Tensor, weights: torch.Tensor):
        '''
        feature: B x C x H x W
        dynamic weights: N x C (N_landmarks x weight_vector_dim)
        '''
        B, C, H, W = feature.shape
        N, _ = weights.shape
        # weights_np = weights.cpu().detach().numpy()
        # w2_np = self.conv.weight.data.squeeze().cpu().detach().numpy()
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1, eps=1e-12)  # N x C
            feature = torch.nn.functional.normalize(feature, p=2, dim=1, eps=1e-12)  # B x C x H x W
        weights_ext = weights.unsqueeze(dim=0).expand(B, N, C)  # B x N x C
        out = torch.bmm(weights_ext, feature.view(B, C, -1))    # B x C x (H*W)
        out = out.reshape(B, N, H, W)
        if self.normalize_weights == False:  # no normalize will lead the conv value being too big.
            out /= 50

        # if self.normalize_weights:
        #     out = (out + 1) / 2.0  # range -1~1 --> 0~1

        return out

# dynamic localization network whose weights are predicted from weight generator
class LocalizationNetV2(nn.Module):
    def __init__(self, normalize_weights = False):
        super(LocalizationNetV2, self).__init__()
        self.normalize_weights = normalize_weights
        # weight = torch.FloatTensor(2048)  #.fill_(1.0/2048)
        # self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, feature: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor =None, **kwargs):
        '''
        feature: B x C x H x W
        dynamic weights: N x C x h x w (N_landmarks x weight_vector_dim x h x w) or N x C
        '''
        B, C, H, W = feature.shape
        if len(weights.shape) == 2:
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # N x C x 1 x 1
        N, C, h, w = weights.shape
        if self.normalize_weights:
            weights = torch.nn.functional.normalize(weights, p=2, dim=1, eps=1e-12)  # N x C x h x w
            feature = torch.nn.functional.normalize(feature, p=2, dim=1, eps=1e-12)  # B x C x H x W
            padding = h // 2
            out = nn.functional.conv2d(feature, weights, bias=bias, stride=1, padding=padding)
            out /= (h * w)  # scaled convolution
            # out = torch.exp((out - 1) / 0.005)
        else:
            padding = h // 2
            # feature = feature * self.weight.view(1, -1, 1, 1)

            # print(feature.max(), feature.min())  # need to check feature value range, which will affect scaled conv.
            if C == 2048:
                weights /= (h * w * 50)  # scaled convolution: Conv(F, W) / (h * w * 50)
                # no normalize will lead the conv value being too big.
                out = nn.functional.conv2d(feature, weights, bias=bias, stride=1, padding=padding)
            else:
                scale_f = kwargs['scaled_conv_factor']
                # scale_f = int(np.sqrt(C) + 0.5)
                weights /= (h * w * scale_f)
                # no normalize will lead the conv value being too big.
                out = nn.functional.conv2d(feature, weights, bias=bias, stride=1, padding=padding)
                # out = nn.functional.relu(out)  # relu may be important

        return out

def heatmap_normalize_layer(pred, normalize_type=0):
    # do normalization for pred heatmap to enforce the summation equal to 1.0
    # pred: B x N x H x W (N is number of body part types)
    if normalize_type == 1:  # 'softmax', \sum P(t) = 1
        B, N, H, W = pred.shape
        pred = pred.reshape(B, N, -1)
        pred = torch.softmax(pred, dim=-1)
        pred = pred.reshape(B, N, H, W)
    elif normalize_type == 2:  # 'relu', 0 <= P(t)
        pred = nn.functional.relu(pred)
    elif normalize_type == 3:  # 'sigmoid', 0 <= P(t) <= 1
        pred = nn.functional.sigmoid(pred-3.0)
    elif normalize_type == 4:  # 'tanh', -1 <= P(t) <= 1
        pred = nn.functional.tanh(pred)
    elif normalize_type == 5:  # 'log_softmax'
        B, N, H, W = pred.shape
        pred = pred.reshape(B, N, -1)
        pred = torch.log_softmax(pred, dim=-1)
        pred = pred.reshape(B, N, H, W)
    elif normalize_type == 6:  # (x + sqrt(x^2+4)) / 2
        pred = (pred + torch.sqrt(pred ** 2 + 0.0004)) / 2
    else:  # is_normalize == 0
        pass
    return pred  # B x N x H x W

class LWModelBase(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LWModelBase, self).__init__()
        self.cfg = cfg
        # encoder related parameters
        trunk                  = cfg.MODEL.ENCODER.TRUNK
        layer_to_freezing      = cfg.MODEL.ENCODER.LAYER_TO_FREEZING
        downsize_factor        = cfg.MODEL.ENCODER.DOWNSIZE_FACTOR
        specify_output_channel = cfg.MODEL.ENCODER.SPECIFY_OUTPUT_CHANNEL
        # ln related parameters
        num_base          = cfg.DATASET[cfg.DATASET.TYPE]['NUM_TRAIN_KP']
        kd_normalize_weights = cfg.MODEL.LWMODELBASE.KD_NORMALIZE_WEIGHTS

        self.encoder = Encoder(trunk, layer_to_freezing, downsize_factor, specify_output_channel)
        feature_channel = self.encoder.feature_dim
        self.ln = LocalizationNetBase(num_base, feature_channel, kd_normalize_weights)

    def forward(self, x):
        '''
        input arguments
        x (image): B x C x H x W

        output arguments
        x (image): B x num_base kps x h x w
        '''
        x = self.encoder(x)
        x = self.ln(x)
        # post-process for heatmaps_predict (different loss function has different requirements)
        if self.cfg.LOSS.TYPE == 'MSE' and self.cfg.LOSS.MSE.CLAMP_HEATMAP:  # 'normalize-mse'
            x = torch.clamp(x, 0, 1)
        elif self.cfg.LOSS.TYPE == 'sigmoid-bce':
            x = torch.sigmoid(x)
        elif self.cfg.LOSS.TYPE == 'GM_GM_L2':
            x = heatmap_normalize_layer(x, self.cfg.LOSS.GM_GM_L2.PRED_PROB_NORM)
        else:
            pass

        return x



class LinearDiag(nn.Module):
    def __init__(self, feature_channel, init_value=1./1000, bias=False):
        super(LinearDiag, self).__init__()
        # weight = torch.FloatTensor(num_features).fill_(1.) # initialize to the identity transform
        weight = torch.FloatTensor(feature_channel).fill_(init_value) # tensor size: C, initialize to the identity transform
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(feature_channel).fill_(0)  # tensor size: C (feature_channel)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        '''
        X: N (N features) x C (feature_channel)
        '''
        assert(X.dim()==2 and X.size(1)==self.weight.size(0))
        out = X * self.weight.expand_as(X)  # channel-wise multiplication (x \odot w)
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out

class LinearMat(nn.Module):
    def __init__(self, feature_channel, init_value=1./1000, bias=True, num_layers=1):
        super(LinearMat, self).__init__()
        if num_layers == 1:
            self.fc_blk = nn.Sequential()
            self.fc_blk.add_module('fc', nn.Linear(feature_channel, feature_channel, bias=bias))
            # self.fc_blk[0].weight.data.copy_(torch.eye(feature_channel, feature_channel) + torch.randn(feature_channel, feature_channel)*0.001)
            self.fc_blk[0].weight.data.copy_(torch.eye(feature_channel, feature_channel) * init_value + torch.randn(feature_channel, feature_channel) * init_value * 0.001)
            # self.fc.bias.data.fill_(0)
            self.fc_blk[0].bias.data.zero_()
        elif num_layers > 1:
            self.fc_blk = nn.Sequential()
            for i in range(num_layers-1):
                self.fc_blk.add_module('fc{}'.format(i), nn.Linear(feature_channel, feature_channel, bias=bias))
                self.fc_blk.add_module('relu{}'.format(i), nn.ReLU())
            self.fc_blk.add_module('fc{}'.format(num_layers-1), nn.Linear(feature_channel, feature_channel, bias=bias))
            # initialization
            for i in range(num_layers-1):
                self.fc_blk[i].weight.data.normal_(mean=0, std=0.01)
                self.fc_blk[i].bias.data.zero_()
            self.fc_blk[(num_layers-1)*2].weight.data.copy_(torch.eye(feature_channel, feature_channel) * init_value + torch.randn(feature_channel, feature_channel) * init_value * 0.001)
            self.fc_blk[(num_layers-1)*2].bias.data.zero_()

    def forward(self, X: torch.Tensor) -> "predicted parameters":
        '''
        X: N (N features) x C (feature_channel)
        '''
        C = self.fc_blk[0].weight.data.shape[0]
        assert X.dim() == 2 and X.size(1) == C

        out = self.fc_blk(X)  # matrix multiplication (x * w)
        # out /= C  # normalize
        return out

class WGByConv(nn.Module):
    def __init__(self, in_channels=2048, out_weights_reso=(1,), generate_bias=False, shared_net_multi_reso=False):
        '''
        the num_layers should >= 3 (depthwise transpose conv-relu + conv1x1-relu + conv1x1)
        '''
        super(WGByConv, self).__init__()
        self.out_weights_reso = out_weights_reso
        self.num_resos = len(out_weights_reso)
        self.shared_net_multi_reso = shared_net_multi_reso
        assert self.num_resos >= 1, 'num_resos should >= 1.'

        # construct the neural network to fit a bank of filters:
        self.weight_net_part1 = nn.ModuleList()
        self.weight_net_part2 = nn.ModuleList()
        if generate_bias:
            self.bias_net = nn.ModuleList()
        else:
            self.bias_net = None

        for i in range(self.num_resos):
            reso = out_weights_reso[i]

            # ------------------------
            # generate weights net
            # first layer: TransposeConv to increase spatial resolution
            if reso == 1:
                net_part1 = nn.Identity()  # identity placeholder
            else:
                net_part1 = nn.Sequential(
                    # nn.Upsample(scale_factor=reso, mode='bilinear'),
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=reso, stride=1, padding=0,
                                       groups=in_channels, bias=True),
                    nn.ReLU()
                )
                # init transposed convolution weightsS
                net_part1[0].weight.data.normal_(mean=1.0, std=0.001)
                net_part1[0].bias.data.zero_()
            self.weight_net_part1.append(net_part1)

            if shared_net_multi_reso and i >= 1:  # if shared, only 1 shared net_part2 for index=0
                pass
            else:
                net_part2 = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)  # 1 x 1 Conv to recover channel
                )
                net_part2[0].weight.data.normal_(mean=0, std=0.01)
                net_part2[0].bias.data.zero_()
                weights_init = torch.eye(in_channels) * 1.0 + torch.randn(in_channels, in_channels) * 1.0 * 0.001
                net_part2[-1].weight.data.copy_(weights_init.reshape(in_channels, in_channels, 1, 1))
                net_part2[-1].bias.data.zero_()
                self.weight_net_part2.append(net_part2)

            # ------------------------
            # generate bias net
            if generate_bias:
                per_bias_net = nn.Sequential(
                    nn.Linear(in_channels, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
                self.bias_net.append(per_bias_net)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, mean=0., std=.005, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        '''
        x: N (N features) x C (feature_channel)
        return: generated weights out_list, each element is a resolution-specific weight N x C x reso x reso
                generated bias_list, [None] or a list with each scalar element
        '''
        weights_list = []
        bias_list = []
        for i in range(self.num_resos):
            weights = self.weight_net_part1[i](x.unsqueeze(-1).unsqueeze(-1))  # N x C x 1 x 1 --> N x C x reso x reso
            if self.shared_net_multi_reso == True:
                weights = self.weight_net_part2[0](weights)
            else:
                weights = self.weight_net_part2[i](weights)
            weights_list.append(weights)
            if self.bias_net is not None:
                bias = self.bias_net[i](x)  # N x C --> N x 1
                bias_list.append(bias.squeeze())  # N x 1 --> N
            else:
                bias_list.append(None)
        return weights_list, bias_list

class WGByConv_ablation_no_CRM(nn.Module):
    def __init__(self, in_channels=2048, out_weights_reso=(1,), generate_bias=False, shared_net_multi_reso=False):
        '''
        the num_layers should >= 3 (depthwise transpose conv-relu + conv1x1-relu + conv1x1)
        '''
        super(WGByConv_ablation_no_CRM, self).__init__()
        self.out_weights_reso = out_weights_reso
        self.num_resos = len(out_weights_reso)
        assert self.num_resos >= 1, 'num_resos should >= 1.'

        # construct the neural network to fit a bank of filters:
        self.weight_net_part1 = nn.ModuleList()
        if generate_bias:
            self.bias_net = nn.ModuleList()
        else:
            self.bias_net = None

        for i in range(self.num_resos):
            reso = out_weights_reso[i]

            # ------------------------
            # generate weights net
            # first layer: TransposeConv to increase spatial resolution
            if reso == 1:
                net_part1 = nn.Identity()  # identity placeholder
            else:
                net_part1 = nn.Sequential(
                    # nn.Upsample(scale_factor=reso, mode='bilinear'),
                    nn.ConvTranspose2d(in_channels, in_channels, kernel_size=reso, stride=1, padding=0,
                                       groups=in_channels, bias=True),
                    # nn.ReLU(inplace=False)
                )
                # init transposed convolution weightsS
                net_part1[0].weight.data.normal_(mean=1.0, std=0.001)
                net_part1[0].bias.data.zero_()
            self.weight_net_part1.append(net_part1)

            # ------------------------
            # generate bias net
            if generate_bias:
                per_bias_net = nn.Sequential(
                    nn.Linear(in_channels, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
                self.bias_net.append(per_bias_net)

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, mean=0., std=.005, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        '''
        x: N (N features) x C (feature_channel)
        return: generated weights out_list, each element is a resolution-specific weight N x C x reso x reso
                generated bias_list, [None] or a list with each scalar element
        '''
        weights_list = []
        bias_list = []
        for i in range(self.num_resos):
            weights = self.weight_net_part1[i](x.unsqueeze(-1).unsqueeze(-1))  # N x C x 1 x 1 --> N x C x reso x reso
            weights_list.append(weights)
            if self.bias_net is not None:
                bias = self.bias_net[i](x)  # N x C --> N x 1
                bias_list.append(bias.squeeze())  # N x 1 --> N
            else:
                bias_list.append(None)
        return weights_list, bias_list

class AttentionBasedBlock(nn.Module):
    def __init__(self, feature_channel, num_base, scale_att=None):
        super(AttentionBasedBlock, self).__init__()
        self.feature_channel = feature_channel
        self.query_layer = nn.Linear(feature_channel, feature_channel)
        self.query_layer.weight.data.copy_(torch.eye(feature_channel, feature_channel) + torch.randn(feature_channel, feature_channel) * 0.001)
        self.query_layer.bias.data.zero_()

        self.num_base = num_base
        # keys = torch.FloatTensor(num_base, feature_channel).normal_(0., np.sqrt(2.0 / feature_channel))
        # self.learnable_keys = nn.Parameter(keys, requires_grad=True)
        self.key_layer = nn.Linear(feature_channel, feature_channel)
        self.key_layer.weight.data.copy_(torch.eye(feature_channel, feature_channel) + torch.randn(feature_channel, feature_channel) * 0.001)
        self.key_layer.bias.data.zero_()

        self.use_scale_att = False
        if scale_att is not None:
            self.use_scale_att = True
            self.scale_att = nn.Parameter(torch.FloatTensor(1).fill_(scale_att), requires_grad=True)  # one learnable scalar

    def forward(self, features, base_weights):
        '''
        features: N (N features) x C (feature_channel) or B (samples) x N (N features) x C (feature_channel)
        base_weights: N_base x C (feature_channel)
        '''
        dim = features.dim()

        if dim == 2:
            N, C = features.shape
            N_base, _ = base_weights.shape
            assert N_base == self.num_base, "The number of base categories triggers error in Class AttentionBasedBlock"

            q = self.query_layer(features)  # N x C
            q_normalize = torch.nn.functional.normalize(q, p=2, dim=1, eps=1e-12)  # N x C
            # keys = self.learnable_keys
            keys = self.key_layer(base_weights)
            keys_normalize = torch.nn.functional.normalize(keys, p=2, dim=1, eps=1e-12)  # N_base x C
            keys_normalize = keys_normalize.transpose(1, 0)
            if self.use_scale_att == False:  # apply learnable temperature
                attention_coefficients = torch.mm(q_normalize, keys_normalize)  # N x N_base
            else:
                attention_coefficients = self.scale_att * torch.mm(q_normalize, keys_normalize)  # N x N_base
            attention_coefficients = torch.softmax(attention_coefficients, dim=1)
            weights_novel = torch.mm(attention_coefficients, base_weights)  # N x C (feature_channel)

        elif dim == 3:
            B, N, C = features.shape
            N_base, _ = base_weights.shape
            assert N_base == self.num_base, "The number of base categories triggers error in Class AttentionBasedBlock"

            features = features.reshape(B * N, C)
            q = self.query_layer(features)
            q = q.reshape(B, N, C)
            q_normalize = torch.nn.functional.normalize(q, p=2, dim=2, eps=1e-12)  # B x N x C
            # keys = self.learnable_keys
            keys = self.key_layer(base_weights)
            keys_normalize = torch.nn.functional.normalize(keys, p=2, dim=1, eps=1e-12)  # N_base x C
            keys_normalize = keys_normalize.transpose(1, 0).unsqueeze(dim=0).repeat(B, 1, 1)  # B x C x N_base
            if self.use_scale_att == False:  # apply learnable temperature
                attention_coefficients = torch.bmm(q_normalize, keys_normalize)  # B x N x N_base
            else:
                attention_coefficients = self.scale_att * torch.bmm(q_normalize, keys_normalize)  # B x N x N_base
            attention_coefficients = torch.softmax(attention_coefficients, dim=attention_coefficients.dim()-1)  # B x N x N_base
            weights_novel = torch.matmul(attention_coefficients, base_weights)  # B x N x C
        else:
            raise ValueError('Unexpected dim in input tensor. Error in AttentionBasedBlock.')

        return weights_novel


class WeightGeneratorWithAtt(nn.Module):
    def __init__(self, feature_channel, linear_type='diag', init_value=1.0/1000,
                 num_base=11, scale_att=None, statistic='individual'):
        '''
        feature_channel, linear_type, init_value are for feature averaging based weight generator,
        feature_channel, num_base, scale_att, statistic are for attention based weight generator.
        statistic: 'individual' or 'mean'
        '''
        super(WeightGeneratorWithAtt, self).__init__()
        # w' = \phi_1 * z_{avg} + \phi_2 * w_{att}
        # w_{att} = E_{i}E_{b} Att(\phi_q * z_i, k_b)*w_b
        if linear_type == 'diag':
            self.wg_feature_avg = LinearDiag(feature_channel, init_value)
            self.linear_att = LinearDiag(feature_channel, init_value)
        elif linear_type == 'mat':
            self.wg_feature_avg = LinearMat(feature_channel, init_value)
            self.linear_att = LinearDiag(feature_channel, init_value)
        else:
            raise ValueError('Wrong linear type for WG. Error in WeightGeneratorWithAtt')

        self.wg_att = AttentionBasedBlock(feature_channel, num_base, scale_att)
        self.statistic = statistic

    def forward(self, base_weights, support_kp_features, support_kp_mask):
        '''
        base_weights: N_base x C (feature_channel)
        support_kp_features:
            it should has size of
            B (samples) x N (N features) x C (feature_channel),
            if self.statistic == 'mean', it means using the avg_features as statistic to get attention-based novel weights;
            else, we use support_kp_features to get attention-based novel weights.
        support_kp_mask: B (samples) x N (N features)
        '''
        # first part of weights is predicted from averaged features (support keypoint prototype)
        # avg_features: N (N features) x C (feature_channel)
        avg_features = average_representations2(support_kp_features.transpose(2, 1), support_kp_mask)  # C x N
        avg_features = avg_features.transpose(1, 0)  # N x C
        weights_novel1 = self.wg_feature_avg(avg_features)

        # second part of weights is predicted based on attention
        if self.statistic == 'mean':
            weights_novel2 = self.wg_att(avg_features, base_weights)  # N x C (feature_channel)
        elif self.statistic == 'individual':
            weights_novel2 = self.wg_att(support_kp_features, base_weights)  # B x N x C (feature_channel)
            weights_novel2 = average_representations2(weights_novel2.transpose(2, 1), support_kp_mask)  # C x N
            weights_novel2 = weights_novel2.transpose(1, 0)  # N x C
        else:
            raise NotImplementedError

        weights_novel = (weights_novel1 + self.linear_att(weights_novel2)) / 2.0

        return weights_novel


class LWModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LWModel, self).__init__()
        self.cfg = cfg
        # encoder related parameters
        trunk                  = cfg.MODEL.ENCODER.TRUNK
        layer_to_freezing      = cfg.MODEL.ENCODER.LAYER_TO_FREEZING
        downsize_factor        = cfg.MODEL.ENCODER.DOWNSIZE_FACTOR
        specify_output_channel = cfg.MODEL.ENCODER.SPECIFY_OUTPUT_CHANNEL
        # wg related parameters
        wg_type   = cfg.MODEL.LWMODEL.WG.TYPE
        statistic = cfg.MODEL.LWMODEL.STATISTIC
        num_base  = cfg.DATASET[cfg.DATASET.TYPE]['NUM_TRAIN_KP']
        scale_att = cfg.MODEL.LWMODEL.SCALE_ATT
        self.normalize_features = cfg.MODEL.LWMODEL.NORMALIZE_FEATURES
        # ln related parameters
        kd_normalize_weights = cfg.MODEL.LWMODEL.KD_NORMALIZE_WEIGHTS


        self.encoder = Encoder(trunk, layer_to_freezing, downsize_factor, specify_output_channel)
        feature_dim = self.encoder.feature_dim

        # wg_type has following options:
        # 'favg-diag': vector learnable parameters for feature-averaging based
        # 'favg-mat': matrix leanable parameters for feature-averaging based
        # 'favg-diag-att': + attention based
        # 'favg-mat-att': + attention based
        self.wg_type = wg_type
        if self.wg_type == 'favg':
            self.weight_generator = None
        elif self.wg_type == 'favg-diag':  # vector learnable parameters
            self.weight_generator = LinearDiag(feature_dim, init_value=1.0, bias=False)  # init_value 1./1000 or 1
        elif self.wg_type =='favg-mat':  # matrix leanable parameters
            fc_blk_num_layers = 1  # number of fc layers: 1
            self.weight_generator = LinearMat(feature_dim, init_value=1.0, num_layers=fc_blk_num_layers)  # init_value 1./1000 or 1
        elif self.wg_type =='favg-mat2':  # matrix leanable parameters
            fc_blk_num_layers = 2  # number of fc layers: 1
            self.weight_generator = LinearMat(feature_dim, init_value=1.0, num_layers=fc_blk_num_layers)  # init_value 1./1000 or 1
        else:  # 'favg-diag-att' or 'favg-mat-att'
            self.statistic = statistic  # 'individual', 'mean'
            if self.statistic not in ['individual', 'mean']:
                raise ValueError('Error for self.statistic.')

            # scale_att = 10.0  # 10.0, None
            if self.wg_type == 'favg-diag-att':
                self.weight_generator = WeightGeneratorWithAtt(feature_dim, linear_type='diag', init_value=1.0,
                                    num_base=num_base, scale_att=scale_att, statistic=statistic)
            elif self.wg_type == 'favg-mat-att':
                self.weight_generator = WeightGeneratorWithAtt(feature_dim, linear_type='mat', init_value=1.0,
                                    num_base=num_base, scale_att=scale_att, statistic=statistic)
            else:
                raise ValueError('Error for self.wg_type in init.')

        if self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO is not None:
            up_scale = self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO
            in_dim = 2048
            self.query_sr = nn.Sequential(
                nn.Upsample(scale_factor=up_scale, mode='bilinear'),
                # nn.Conv2d(in_dim, in_dim, kernel_size=5, stride=1, padding=2),
                # nn.ReLU(),
                # nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.query_sr = None

        # dynamic localization network
        self.ln = LocalizationNet(normalize_weights=kd_normalize_weights)

        # self.base_weights = torch.randn(11, 2048).cuda()
        self.base_weights = None  #torch.FloatTensor(num_base, feature_dim).normal_(0., np.sqrt(2.0 / feature_dim))
        # self.register_buffer('base_weights', base_weights)
        self.need_base_weights = True if 'att' in self.wg_type else False
        print('Note: base weights have not yet been initialized!')

    def init_weights(self, base_model_file='', pretrained=''):
        # 1. load encoder + ln weights from base model
        if self.need_base_weights:
            if os.path.isfile(base_model_file):
                base_model_state_dict = torch.load(base_model_file)
                need_init_state_dict = {}
                for name, m  in base_model_state_dict.items():
                    if 'encoder' in name:
                        need_init_state_dict[name[8:]] = m
                        # need_init_state_dict[name] = m
                # method 1
                self.encoder.load_state_dict(need_init_state_dict)
                # method 2
                # updated_state_dict = self.state_dict()
                # updated_state_dict.update(need_init_state_dict)
                # self.load_state_dict(updated_state_dict)

                self.base_weights = base_model_state_dict['ln.conv.weight']  # num_base x feature_dim x 1 x 1
                assert self.base_weights.shape[1] == self.encoder.feature_dim and self.base_weights.shape[2] == 1 and self.base_weights.shape[3] == 1
                num_base, feature_dim = self.base_weights.shape[:2]
                self.base_weights = self.base_weights.reshape(num_base, feature_dim)
                print('==> loading base model {}.'.format(base_model_file))
            else:
                print('Error: base_model_file is not available! Please check cfg.MODEL.BASE_MODEL_FILE')
                raise ValueError

        # 2. load encoder + wg weights from pretrained model
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_state_dict)
            print('==> loading pretrained model {}.'.format(pretrained))

    def forward(self, supports, queries, support_kps, support_kp_mask, **kwargs):
        '''
        input arguments
        supports (image): B1 x 3 x H x W
        queries (image): B2 x 3 x H x W
        support_kps: B1 x N x 2
        support_kp_mask: B1 x N

        output arguments
        heatmaps_predict: B2 x N x H x W

        '''
        support_features = self.encoder(x=supports)  # B1 x C x H x W
        query_features = self.encoder(x=queries)     # B2 x C x H x W

        # B1 x C x N
        context_mode = 'soft_fiber_gaussian'  # 'soft_fiber_bilinear'
        sigma           = self.cfg.DATASET.SIGMA
        downsize_factor = self.cfg.MODEL.ENCODER.DOWNSIZE_FACTOR
        image_length    = self.cfg.DATASET.SQUARE_IMAGE_LENGTH
        support_repres, conv_query_features = extract_representations(support_features, support_kps, support_kp_mask, context_mode=context_mode,
                        sigma=sigma, downsize_factor=downsize_factor, image_length=image_length, together_trans_features=None)

        '''Either normalize + l2 loss (mse) or sigmoid + bec loss is better for WG based FSL, according to my 
        experiments. Using normalization is like to do cosine-similarity scores, and this operation has good 
        generalization to novel categories only when there is 1 linear layer. When the classifier has multiple layers, 
        it doesn't have better performance and even have drops.
        '''
        # Weight Generator (WG)
        base_weights = self.base_weights
        if self.normalize_features:
            support_repres = nn.functional.normalize(support_repres, p=2, dim=1, eps=1e-12)
            if base_weights is not None:
                base_weights = nn.functional.normalize(base_weights, p=2, dim=1, eps=1e-12)
        if self.wg_type == 'favg':
            avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
            weights = avg_support_repres.transpose(1, 0)
        elif self.wg_type == 'favg-diag' or self.wg_type == 'favg-mat' or self.wg_type == 'favg-mat2':
            avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
            weights = self.weight_generator(avg_support_repres.transpose(1, 0))  # N x C
        elif self.wg_type == 'favg-diag-att' or self.wg_type == 'favg-mat-att':
            # base_weights: N_{base} x C
            # support_repres: B1 x C x N
            weights = self.weight_generator(base_weights, support_repres.transpose(2, 1), support_kp_mask)  # N x C
        else:
            raise NotImplementedError

        if self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO is not None:
            query_features = self.query_sr(query_features)

        # the parameters of localization_net is injected by ``weights''
        heatmaps_predict = self.ln(query_features, weights)  # B2 x N x H x W
        # post-process for heatmaps_predict (different loss function has different requirements)
        if self.cfg.LOSS.TYPE == 'MSE' and self.cfg.LOSS.MSE.CLAMP_HEATMAP:  # 'normalize-mse'
            heatmaps_predict = torch.clamp(heatmaps_predict, 0, 1)
        elif self.cfg.LOSS.TYPE == 'sigmoid-bce':
            heatmaps_predict = torch.sigmoid(heatmaps_predict)
        elif self.cfg.LOSS.TYPE == 'GM_GM_L2':
            heatmaps_predict = heatmap_normalize_layer(heatmaps_predict, self.cfg.LOSS.GM_GM_L2.PRED_PROB_NORM)
        else:
            pass

        return heatmaps_predict

class LWModelMultiScale(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LWModelMultiScale, self).__init__()
        self.cfg = cfg
        # encoder related parameters
        trunk = cfg.MODEL.ENCODER.TRUNK
        self.trunk = trunk
        if trunk == 'RESNET50':
            layer_to_grab_features = cfg.MODEL.ENCODER[trunk].LAYER_TO_GRAB_FEATURES
            layer_to_freezing      = cfg.MODEL.ENCODER[trunk].LAYER_TO_FREEZING
            downsize_factor_list   = cfg.MODEL.ENCODER[trunk].DOWNSIZE_FACTOR
            specify_output_channel = cfg.MODEL.ENCODER[trunk].SPECIFY_OUTPUT_CHANNEL
            self.encoder = EncoderMultiScale(trunk, layer_to_freezing, downsize_factor_list, layer_to_grab_features=layer_to_grab_features)
            feature_dims = self.encoder.feature_dims
        elif trunk == 'DINO':
            arch = cfg.MODEL.ENCODER[trunk].ARCH
            patch_size = cfg.MODEL.ENCODER[trunk].PATCH_SIZE
            dino_pretrained_path = cfg.MODEL.ENCODER[trunk].PRETRAINED_PATH
            layer_to_freezing = cfg.MODEL.ENCODER[trunk].LAYER_TO_FREEZING
            downsize_factor_list = cfg.MODEL.ENCODER[trunk].DOWNSIZE_FACTOR

            self.encoder = DINO_VIT.__dict__[arch](patch_size=patch_size, frozen_layer=layer_to_freezing)  # define model
            state_dict = torch.load(dino_pretrained_path, map_location="cpu")
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print('DINO pretrained weights found at {} and loaded with msg: {}'.format(dino_pretrained_path, msg))

            feature_dims = [self.encoder.num_features] * len(downsize_factor_list)
        else:
            raise NotImplementedError
        # every type of encoder should have ``downsize_factor_list"
        self.downsize_factor_list = downsize_factor_list


        # wg related parameters
        wg_type   = cfg.MODEL.LWMODEL.WG.TYPE
        statistic = cfg.MODEL.LWMODEL.STATISTIC
        num_base  = cfg.DATASET[cfg.DATASET.TYPE]['NUM_TRAIN_KP']
        scale_att = cfg.MODEL.LWMODEL.SCALE_ATT
        self.normalize_features = cfg.MODEL.LWMODEL.NORMALIZE_FEATURES
        # ln related parameters
        kd_normalize_weights = cfg.MODEL.LWMODEL.KD_NORMALIZE_WEIGHTS

        # wg_type has following options:
        # 'favg-diag': vector learnable parameters for feature-averaging based
        # 'favg-mat': matrix leanable parameters for feature-averaging based
        # 'favg-diag-att': + attention based
        # 'favg-mat-att': + attention based
        num_scales = len(downsize_factor_list)
        self.wg_type = wg_type
        self.weight_generator_list = nn.ModuleList()
        for i in range(num_scales):
            feature_dim = feature_dims[i]
            if self.wg_type == 'favg':
                weight_generator = None
            elif self.wg_type == 'favg-diag':  # vector learnable parameters
                weight_generator = LinearDiag(feature_dim, init_value=1.0, bias=False)  # init_value 1./1000 or 1
            elif self.wg_type =='favg-mat':  # matrix leanable parameters
                fc_blk_num_layers = 1  # number of fc layers: 1
                weight_generator = LinearMat(feature_dim, init_value=1.0, num_layers=fc_blk_num_layers)  # init_value 1./1000 or 1
            elif self.wg_type =='favg-mat2':  # matrix leanable parameters
                fc_blk_num_layers = 2  # number of fc layers: 1
                weight_generator = LinearMat(feature_dim, init_value=1.0, num_layers=fc_blk_num_layers)  # init_value 1./1000 or 1
            elif self.wg_type == 'favg-conv2':  # matrix leanable parameters
                wg_out_weights_reso = cfg.MODEL.LWMODEL.WG.OUT_WEIGHTS_RESO
                wg_generate_bias = cfg.MODEL.LWMODEL.WG.GENERATE_BIAS
                wg_shared_net_multi_reso = cfg.MODEL.LWMODEL.WG.SHARED_NET_MULTI_RESO
                weight_generator = WGByConv(feature_dim, wg_out_weights_reso,
                                            generate_bias=wg_generate_bias,
                                            shared_net_multi_reso=wg_shared_net_multi_reso
                                            )
            elif self.wg_type == 'favg-conv2-ablation':  # no CRM (channel refinement module)
                wg_out_weights_reso = cfg.MODEL.LWMODEL.WG.OUT_WEIGHTS_RESO
                wg_generate_bias = cfg.MODEL.LWMODEL.WG.GENERATE_BIAS
                wg_shared_net_multi_reso = cfg.MODEL.LWMODEL.WG.SHARED_NET_MULTI_RESO
                weight_generator = WGByConv_ablation_no_CRM(feature_dim, wg_out_weights_reso,
                                            generate_bias=wg_generate_bias,
                                            )
            else:  # 'favg-diag-att' or 'favg-mat-att'
                self.statistic = statistic  # 'individual', 'mean'
                if self.statistic not in ['individual', 'mean']:
                    raise ValueError('Error for self.statistic.')

                # scale_att = 10.0  # 10.0, None
                if self.wg_type == 'favg-diag-att':
                    weight_generator = WeightGeneratorWithAtt(feature_dim, linear_type='diag', init_value=1.0,
                                        num_base=num_base, scale_att=scale_att, statistic=statistic)
                elif self.wg_type == 'favg-mat-att':
                    weight_generator = WeightGeneratorWithAtt(feature_dim, linear_type='mat', init_value=1.0,
                                        num_base=num_base, scale_att=scale_att, statistic=statistic)
                else:
                    raise ValueError('Error for self.wg_type in init.')
            self.weight_generator_list.append(weight_generator)

        # query feature up-sampling
        if self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO is not None:
            self.query_sr_list = nn.ModuleList()
            up_scales_list = self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO
            assert len(up_scales_list) == num_scales, 'the number of scales should match.'
            for up_scale in up_scales_list:
                if up_scale == 1.0:
                    up_sr = nn.ReLU()  # used as placeholder (has no effect in inference)
                else:
                    up_sr = nn.Upsample(scale_factor=up_scale, mode='bilinear')
                self.query_sr_list.append(up_sr)
        else:
            self.query_sr_list = None

        # dynamic localization network
        self.ln = LocalizationNetV2(normalize_weights=kd_normalize_weights)

        # output heatmap up-sampling
        if cfg.MODEL.LWMODEL.KD_SUPER_RESO.TYPE is not None:
            self.output_sr_list = nn.ModuleList()
            output_up_type = cfg.MODEL.LWMODEL.KD_SUPER_RESO.TYPE
            output_up_scales = cfg.MODEL.LWMODEL.KD_SUPER_RESO.UP_SCALES
            for i, up_scale in enumerate(output_up_scales):
                if output_up_type == 'bilinear':
                    up_sr = nn.Upsample(scale_factor=up_scale, mode='bilinear') if up_scale != 1.0 else nn.Identity()
                elif output_up_type == 'conv1':
                    up_sr = nn.Sequential(
                        nn.Upsample(scale_factor=up_scale, mode='bilinear'),
                        nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
                    )
                elif output_up_type == 'conv2':
                    up_sr = nn.Sequential(
                        nn.Upsample(scale_factor=up_scale, mode='bilinear'),
                        nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),  # single-channel 5x5 convolution
                        nn.ReLU(),
                        nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
                    )
                    # up_sr[1].weight.data.normal_(mean=1 / 5 ** 2, std=0.001)
                    # up_sr[1].bias.data.zero_()
                    # up_sr[3].weight.data.normal_(mean=1 / 5 ** 2, std=0.001)
                    # up_sr[3].bias.data.zero_()
                elif output_up_type in ['fc1', 'fc2']:
                    image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH
                    dz = downsize_factor_list[i]
                    reso_per_scale = image_length // dz
                    fc_in_ch = reso_per_scale ** 2
                    fc_out_ch = (reso_per_scale * int(up_scale)) ** 2
                    if output_up_type == 'fc1':
                        up_sr = nn.Sequential(
                            nn.Linear(fc_in_ch, fc_out_ch)
                        )
                    elif output_up_type == 'fc2':
                        up_sr = nn.Sequential(
                            nn.Linear(fc_in_ch, fc_out_ch),
                            nn.ReLU(),
                            nn.Linear(fc_out_ch, fc_out_ch)
                        )
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                self.output_sr_list.append(up_sr)
        else:
            self.output_sr_list = None

        # # Transform output heatmap to coordinates -1.0 <= (x, y) <= 1.0
        # if cfg.MODEL.LWMODEL.HEATMAP_TO_COORD.TYPE is not None:
        #     self.heatmap_to_coord_type = cfg.MODEL.LWMODEL.HEATMAP_TO_COORD.TYPE
        #     scale_factor = cfg.MODEL.LWMODEL.HEATMAP_TO_COORD.SCALE_FACTOR
        #     self.heatmap_to_coord_func_list = []
        #     output_up_scales = cfg.MODEL.LWMODEL.KD_SUPER_RESO.UP_SCALES if cfg.MODEL.LWMODEL.KD_SUPER_RESO.TYPE != None else \
        #         [1.0] * num_scales
        #     for i in range(num_scales):
        #         image_length = cfg.DATASET.SQUARE_IMAGE_LENGTH
        #         dz = downsize_factor_list[i]
        #         up_scale = output_up_scales[i]
        #         new_length = image_length // dz * int(up_scale)
        #         if self.heatmap_to_coord_type == 'soft-argmax':
        #             hm_to_coord_func = SoftArgmax(new_length, new_length, scale_factor)
        #         self.heatmap_to_coord_func_list.append(hm_to_coord_func)
        # else:
        #     self.heatmap_to_coord_type = None

        # self.base_weights = torch.randn(11, 2048).cuda()
        self.base_weights = None  #torch.FloatTensor(num_base, feature_dim).normal_(0., np.sqrt(2.0 / feature_dim))
        # self.register_buffer('base_weights', base_weights)
        self.need_base_weights = True if 'att' in self.wg_type else False
        print('Note: base weights have not yet been initialized!')

        # init cost evaluator
        self.set_cost_eval(eval_cost=False)  # by default disabled, but it can be opened manually

    def init_weights(self, base_model_file='', pretrained=''):
        # 1. load encoder + ln weights from base model
        if self.need_base_weights:
            if os.path.isfile(base_model_file):
                base_model_state_dict = torch.load(base_model_file)
                need_init_state_dict = {}
                for name, m  in base_model_state_dict.items():
                    if 'encoder' in name:
                        need_init_state_dict[name[8:]] = m
                        # need_init_state_dict[name] = m
                # method 1
                self.encoder.load_state_dict(need_init_state_dict)
                # method 2
                # updated_state_dict = self.state_dict()
                # updated_state_dict.update(need_init_state_dict)
                # self.load_state_dict(updated_state_dict)

                self.base_weights = base_model_state_dict['ln.conv.weight']  # num_base x feature_dim x 1 x 1
                assert self.base_weights.shape[1] == self.encoder.feature_dim and self.base_weights.shape[2] == 1 and self.base_weights.shape[3] == 1
                num_base, feature_dim = self.base_weights.shape[:2]
                self.base_weights = self.base_weights.reshape(num_base, feature_dim)
                print('==> loading base model {}.'.format(base_model_file))
            else:
                print('Error: base_model_file is not available! Please check cfg.MODEL.BASE_MODEL_FILE')
                raise ValueError

        # 2. load encoder + wg weights from pretrained model
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_state_dict)
            print('==> loading pretrained model {}.'.format(pretrained))

    def heatmap_upscale(self, heatmaps_list):
        '''
        a list of num_scale heatmaps, each heatmap is S x B x N x H x W
        return a list of num_scale heatmaps with each S x B x N x H x W
        '''
        # output fusion via up-sampling
        output_up_type = self.cfg.MODEL.LWMODEL.KD_SUPER_RESO.TYPE
        output_up_scales = self.cfg.MODEL.LWMODEL.KD_SUPER_RESO.UP_SCALES  # pre-defined up-scales
        assert len(output_up_scales) == len(heatmaps_list), 'the number of groups should match.'

        heatmaps_list_new = []
        for i, up_scale in enumerate(output_up_scales):
            heatmaps = heatmaps_list[i]  # S x B x N x H x W
            S, B, N, H, W = heatmaps.shape
            if output_up_type == 'bilinear':
                heatmaps = heatmaps.reshape(S*B, N, H, W)  # (S*B) x N x H x W
                if up_scale == 1.0:
                    heatmap_up = heatmaps
                else:
                    heatmap_up = self.output_sr_list[i](heatmaps)
            elif output_up_type in ['conv1', 'conv2']:
                tmp = heatmaps.reshape(S*B*N, 1, H, W)
                heatmap_up = self.output_sr_list[i](tmp)
            elif output_up_type in ['fc1', 'fc2']:
                tmp = heatmaps.reshape(S*B*N, H*W)
                heatmap_up = self.output_sr_list[i](tmp)  # (S*B*N) x (H2*W2)
            else:
                raise NotImplementedError

            H2, W2 = int(H * up_scale), int(W * up_scale)
            heatmap_up = heatmap_up.reshape(S, B, N, H2, W2)
            heatmaps_list_new.append(heatmap_up)

        return heatmaps_list_new

    def output_fuse(self, outputs_list, fuse_method='avg'):
        '''
        a list of num_scale outputs, each output is B x N x H x W or B x N x 2
        fuse_method: 'avg' or 'prod'
        return outputs_fused: B x N x H x W or B x N x 2
        '''
        assert outputs_list[0].shape[-1] == outputs_list[-1].shape[-1], 'resolution need to be same!'
        if fuse_method == 'avg':
            outputs_fused = torch.mean(torch.stack(outputs_list), dim=0)
        else:  # 'prod'
            outputs_fused = torch.prod(torch.stack(outputs_list), dim=0)
        return outputs_fused

    def heatmap_to_coord(self, heatmaps_list):
        '''
        a list of num_scale heatmaps, each heatmap is B x N x H x W
        method: 'soft-argmax'
        return coords_list whose each element has size of B x N x 2 (-1.0 ~ 1.0).
        '''
        coords_list = []
        for i in range(len(heatmaps_list)):
            heatmap = heatmaps_list[i]  # B x N x H x W
            coords = self.heatmap_to_coord_func_list[i](heatmap)
            coords_list.append(coords)
        return coords_list

    def domain_alignment(self, align_repres, align_kps_mask, alignment_type, sim_type, tau,
                         sampled_neg_repres=None):
        '''
        align_repres:   (S * (B1+B2)) x C x N_align (use_proto=False) or S x C x N_align (use_proto=True)
        align_kps_mask: (S * (B1+B2)) x N_align     (use_proto=False) or S x N_align     (use_proto=True)
        tau: temperature
        sampled_neg_repres: (S * (B1+B2)) x C x N_sample (use_proto=False) or S x C x ((B1+B2)*N_sample) (use_proto=True)
        '''
        num_samples, N_align = align_kps_mask.shape
        C_f = align_repres.shape[1]
        union_align_kps_mask = align_kps_mask.sum(0)  # N_align

        # ---------------------------------------------------------------------------------------------
        # use sampled negative points to enhance contrastive
        use_sampled_neg = True if sampled_neg_repres is not None else False
        if use_sampled_neg == True:
            use_all_sampled_negs = self.cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.USE_ALL_NEG
            if use_all_sampled_negs == True:  # True, combine all sampleds negs
                sampled_neg_repres_tmp = sampled_neg_repres.permute(1, 0, 2).flatten(1)  # C x N_other
            else:
                sampled_neg_repres_tmp = None
        # ---------------------------------------------------------------------------------------------

        if alignment_type == 'align_repres':
            use_sim_or_dist = 1 if sim_type in ['cosine', 'rbf'] else 2
            align_loss = 0
            num_pairs = 0
            for n in range(N_align):
                if union_align_kps_mask[n] < 2:  # at least two samples
                    continue
                repres_tmp = align_repres[:, :, n]  # S x C
                mask_tmp = align_kps_mask[:, n]     # S
                mask_matrix = mask_tmp.unsqueeze(-1) * mask_tmp.unsqueeze(0)
                num_pairs_tmp = (mask_matrix.sum() - num_samples) / 2.0  # remove diagonal as they are ones (use sim) or zeros (use dist)
                if use_sim_or_dist == 1:  # use similarity
                    sim_matrix = compute_similarity(self.cfg, repres_tmp, repres_tmp, sim_type)  # S x S
                    align_loss_tmp = ((sim_matrix - 1) ** 2 * mask_matrix).sum() / 2.0  # GT is all-one matrix since they are all positive pairs
                else:  # use distance
                    dist_matrix = compute_distance(repres_tmp, repres_tmp, sim_type)  # S x S
                    align_loss_tmp = (dist_matrix * mask_matrix).sum() / 2.0
                align_loss += align_loss_tmp
                num_pairs += num_pairs_tmp
            align_loss = (align_loss/num_pairs) if num_pairs > 0 else torch.tensor(0.).cuda()
        elif alignment_type == 'align_relation':  # pointwise align
            align_relation_type = self.cfg.LOSS.DOMAIN_ALIGNMENT.ALIGN_RELATION.TYPE
            if align_relation_type == 'KLD':
                loss_func = nn.KLDivLoss(reduction="sum", log_target=False)
            align_loss = 0
            num_pairs = 0
            num_kps_per_sample = align_kps_mask.sum(-1)
            for s1 in range(num_samples):
                for s2 in range(s1+1, num_samples):
                    if num_kps_per_sample[s1] <= 0 or num_kps_per_sample[s2] <= 0:
                        continue
                    mask_tmp1 = align_kps_mask[s1]  # N_align
                    mask_tmp2 = align_kps_mask[s2]  # N_align
                    mask_combine = (mask_tmp1 * mask_tmp2).long()
                    if torch.sum(mask_combine) == 0:  # must exist one-one valid kp pair. thus, sim_matrix's diagonal not all zeros.
                        continue
                    mask_matrix = mask_tmp1.unsqueeze(-1) * mask_tmp2.unsqueeze(0)
                    repres_tmp1 = align_repres[s1]  # C x N_align
                    repres_tmp2 = align_repres[s2]  # C x N_align
                    sim_matrix = compute_similarity(self.cfg, repres_tmp1.T, repres_tmp2.T, sim_type)  # N_align x N_align
                    groundtruth = torch.eye(N_align).cuda()
                    if align_relation_type == 'MSE':  # pointwise MSE
                        align_loss_tmp = ((sim_matrix - groundtruth) ** 2 * mask_matrix).sum()
                    elif align_relation_type == 'KLD':  # pointwise KLD
                        sim_matrix = sim_matrix - 100 * (1 - mask_matrix)  # masking
                        sim_matrix /= tau  # temperature
                        log_prob1 = nn.functional.log_softmax(sim_matrix, dim=1)
                        align_loss_tmp1 = loss_func(log_prob1 * mask_matrix, groundtruth * mask_matrix)
                        log_prob2 = nn.functional.log_softmax(sim_matrix.T, dim=1)
                        align_loss_tmp2 = loss_func(log_prob2 * mask_matrix.T, groundtruth * mask_matrix.T)
                        align_loss_tmp = (align_loss_tmp1 + align_loss_tmp2) / 2
                    else:
                        raise NotImplementedError
                    num_pairs_tmp = mask_matrix.sum()
                    align_loss += (align_loss_tmp / num_pairs_tmp)
                    num_pairs += 1  # here means a pair of images, not a pair of kps, as we already perform avg
            align_loss = (align_loss / num_pairs) if num_pairs > 0 else torch.tensor(0.).cuda()
        elif alignment_type == 'align_relation2':  # relation contrasive loss
            use_noise= self.cfg.LOSS.DOMAIN_ALIGNMENT.INJECT_NOISE.USE
            noise_type = self.cfg.LOSS.DOMAIN_ALIGNMENT.INJECT_NOISE.NOISE_TYPE
            noise_std  = self.cfg.LOSS.DOMAIN_ALIGNMENT.INJECT_NOISE.STD

            align_loss = 0
            num_pairs = 0
            loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
            num_kps_per_sample = align_kps_mask.sum(-1)
            for s1 in range(num_samples):
                for s2 in range(s1 + 1, num_samples):
                    if num_kps_per_sample[s1] <= 0 or num_kps_per_sample[s2] <= 0:
                        continue
                    mask_tmp1 = align_kps_mask[s1]  # N_align
                    mask_tmp2 = align_kps_mask[s2]  # N_align
                    mask_combine = (mask_tmp1 * mask_tmp2).long()
                    if torch.sum(mask_combine) == 0:  # must exist one-one valid kp pair. thus, sim_matrix's diagonal not all zeros.
                        continue
                    mask_matrix = mask_tmp1.unsqueeze(-1) * mask_tmp2.unsqueeze(0)
                    repres_tmp1 = align_repres[s1]  # C x N_align
                    repres_tmp2 = align_repres[s2]  # C x N_align
                    sim_matrix = compute_similarity(self.cfg, repres_tmp1.T, repres_tmp2.T, sim_type)  # N_align x N_align
                    sim_matrix = sim_matrix - 100 * (1 - mask_matrix)  # masking
                    sim_matrix /= tau  # temperature
                    groundtruth = torch.arange(N_align, dtype=torch.long).cuda()
                    groundtruth = groundtruth * mask_combine  - (1 - mask_combine)  # set invalid kps to be -1

                    #----------------------------
                    # noise
                    if use_noise == False:
                        sim_matrix_T = sim_matrix.T
                    else:  # True, corrupted by noise
                        noise = torch.randn(C_f, N_align).cuda() * noise_std  # noise ~ N(0, std)
                        repres_tmp1_noise = apply_noise(repres_tmp1, noise, type=noise_type)  # C x N_align
                        repres_tmp2_noise = apply_noise(repres_tmp2, noise, type=noise_type)  # C x N_align
                        sim_vector1 = compute_similarity2(self.cfg, repres_tmp1.T, repres_tmp2_noise.T, sim_type)  # N_align
                        sim_vector2 = compute_similarity2(self.cfg, repres_tmp2.T, repres_tmp1_noise.T, sim_type)  # N_align
                        sim_vector1 /= tau
                        sim_vector2 /= tau
                        clear_diag_mask = 1 - torch.eye(N_align).cuda()
                        sim_matrix_diag_clear = sim_matrix * clear_diag_mask
                        sim_matrix = sim_matrix_diag_clear + torch.diag(sim_vector1)
                        sim_matrix_T = sim_matrix_diag_clear.T + torch.diag(sim_vector2)
                    # ----------------------------

                    if use_sampled_neg == False:
                        align_loss_tmp1 = loss_func(sim_matrix, groundtruth)
                        align_loss_tmp2 = loss_func(sim_matrix_T, groundtruth)
                    else:  # use sampled negatives to enhance contrastive learning
                        if use_all_sampled_negs == False:  # False, only used sampled negatives from pairwise image/episodes
                            sampled_neg_repres_tmp1 = sampled_neg_repres[s1]  # C x N_sampled
                            sampled_neg_repres_tmp2 = sampled_neg_repres[s2]  # C x N_sampled
                            sampled_neg_repres_tmp = torch.cat([sampled_neg_repres_tmp1, sampled_neg_repres_tmp2], dim=-1)  # C x N_other
                        other_sim_mat1 = compute_similarity(self.cfg, repres_tmp1.T, sampled_neg_repres_tmp.T, sim_type)  # N_align x N_other
                        other_sim_mat1 = other_sim_mat1 - 100 * (1-mask_tmp1).unsqueeze(-1)  # masking
                        other_sim_mat1 /= tau
                        other_sim_mat2 = compute_similarity(self.cfg, repres_tmp2.T, sampled_neg_repres_tmp.T, sim_type)  # N_align x N_other
                        other_sim_mat2 = other_sim_mat2 - 100 * (1-mask_tmp2).unsqueeze(-1)  # masking
                        other_sim_mat2 /= tau
                        sim_matrix_comb1 = torch.cat([sim_matrix, other_sim_mat1], dim=-1)    # N_align x (N_align + N_other)
                        sim_matrix_comb2 = torch.cat([sim_matrix_T, other_sim_mat2], dim=-1)  # N_align x (N_align + N_other)
                        align_loss_tmp1 = loss_func(sim_matrix_comb1, groundtruth)
                        align_loss_tmp2 = loss_func(sim_matrix_comb2, groundtruth)
                    align_loss_tmp = (align_loss_tmp1 + align_loss_tmp2) / 2
                    align_loss += align_loss_tmp
                    num_pairs += 1  # here means a pair of images, not a pair of kps, as we already perform avg
                    # if np.isnan(align_loss_tmp.cpu().detach().numpy()):
                    #     print('error')
            align_loss = (align_loss / num_pairs) if num_pairs > 0 else torch.tensor(0.).cuda()
        elif alignment_type == 'contrastive':
            # we traverse the anchor point, sample one positive point, many negative points
            align_loss = 0
            num_pairs = 0
            loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
            num_valid_kps_total = torch.sum(union_align_kps_mask)
            for n in range(N_align):
                if union_align_kps_mask[n] < 2:  # at least two pos points
                    continue
                if (num_valid_kps_total - union_align_kps_mask[n]) < 1:  # at least one neg points
                    continue

                pos_repres = align_repres[:, :, n]  # S x C
                pos_mask = align_kps_mask[:, n]  # S
                pos_pool = pos_repres[pos_mask.bool()]  # num_pos x C
                num_pos = pos_pool.shape[0]

                inds = torch.ones(N_align).bool()  # define a mask
                inds[n] = False
                neg_repres = align_repres[:, :, inds]    # S x C x (N_align - 1)
                neg_repres = neg_repres.permute(0, 2, 1).flatten(0, 1) # (S * (N_align - 1)) x C
                neg_mask = align_kps_mask[:, inds]  # S x (N_align - 1)
                neg_mask = neg_mask.reshape(-1).bool()  # (S * (N_align - 1))
                neg_pool = neg_repres[neg_mask]  # num_neg x C
                num_neg = neg_pool.shape[0]
                if self.cfg.LOSS.DOMAIN_ALIGNMENT.CONTRASTIVE.EXHAUST_POS_PAIR == False:
                    for s in range(num_pos):  # traverse each anchor
                        anchor = pos_pool[s]  # C
                        anchor = anchor.unsqueeze(0)  # 1 x C
                        pos_ind = np.random.randint(0, num_pos, size=1)[0]  # sample another positive point
                        pos_tmp = pos_pool[pos_ind]  # C
                        pos_tmp = pos_tmp.unsqueeze(0)  # 1 x C
                        neg_tmp = neg_pool  # take all negatives
                        combine = torch.cat([pos_tmp, neg_tmp], dim=0)  # (1 + num_neg) x C
                        sim_matrix = compute_similarity(self.cfg, anchor, combine, sim_type)  # 1 x (1 + num_neg)
                        sim_matrix = sim_matrix / tau  # temperature
                        groundtruth = torch.arange(1, dtype=torch.long).cuda()
                        align_loss_tmp = loss_func(sim_matrix, groundtruth)
                        align_loss += align_loss_tmp
                        num_pairs += 1  # note we perform "sum"
                else:
                    for s1 in range(num_pos): # traverse all positive pairs
                        for s2 in range(s1 + 1, num_pos):
                            anchor = pos_pool[s1]  # C
                            anchor = anchor.unsqueeze(0)  # 1 x C
                            pos_tmp = pos_pool[s2]
                            pos_tmp = pos_tmp.unsqueeze(0)  # 1 x C
                            neg_tmp = neg_pool  # take all negatives
                            combine = torch.cat([pos_tmp, neg_tmp], dim=0)  # (1 + num_neg) x C
                            sim_matrix = compute_similarity(self.cfg, anchor, combine, sim_type)  # 1 x (1 + num_neg)
                            sim_matrix = sim_matrix / tau  # temperature
                            groundtruth = torch.arange(1, dtype=torch.long).cuda()
                            align_loss_tmp = loss_func(sim_matrix, groundtruth)
                            align_loss += align_loss_tmp
                            num_pairs += 1  # note we perform "sum"

            align_loss = (align_loss / num_pairs) if num_pairs > 0 else torch.tensor(0.).cuda()
        elif alignment_type == 'multispecies':
            loss_func = nn.KLDivLoss(reduction="mean", log_target=False)
            num_valid_kps_total = torch.sum(union_align_kps_mask)
            if num_valid_kps_total >= 2:
                repres_tmp = align_repres.permute(0, 2, 1).reshape(num_samples*N_align, -1)  # (S*N_align) x C
                mask_bool = align_kps_mask.reshape(-1).bool()  # (S*N_align)
                filtered_repres = repres_tmp[mask_bool, :]     # N_filter x C
                sim_matrix = compute_similarity(self.cfg, filtered_repres, filtered_repres, sim_type)  # N_filter x N_filter
                sim_matrix /= tau  # temperature

                # construct groundtruth label
                kp_ind = torch.arange(N_align).unsqueeze(0).repeat(num_samples, 1)  # S x N_align
                kp_ind = kp_ind.reshape(-1)  # (S*N_align)
                filtered_kp_ind = kp_ind[mask_bool]  # N_filter
                groundtruth = (filtered_kp_ind.unsqueeze(1) == filtered_kp_ind.unsqueeze(0))  # N_filter x N_filter
                groundtruth = groundtruth.float().cuda()  # N_filter x N_filter
                # if self.cfg.LOSS.DOMAIN_ALIGNMENT.MULTISPECIES.GT_PROB_AVG == True:  # avg gt prob, will decrease pck
                #     sum_tmp = groundtruth.sum(-1).unsqueeze(-1)  # N_filter x 1
                #     groundtruth /= sum_tmp

                if use_sampled_neg == False:
                    log_prob1 = nn.functional.log_softmax(sim_matrix, dim=1)
                    align_loss_tmp1 = loss_func(log_prob1, groundtruth)
                    log_prob2 = nn.functional.log_softmax(sim_matrix.T, dim=1)
                    align_loss_tmp2 = loss_func(log_prob2, groundtruth)
                    align_loss = (align_loss_tmp1 + align_loss_tmp2) / 2
                else:  # use sampled negatives to enhance contrastive learning
                    if use_all_sampled_negs == False:  # False, only used sampled negatives from pairwise image/episodes
                        (s1, s2) = np.random.randint(0, num_samples, 2)
                        sampled_neg_repres_tmp1 = sampled_neg_repres[s1]  # C x N_sampled
                        sampled_neg_repres_tmp2 = sampled_neg_repres[s2]  # C x N_sampled
                        sampled_neg_repres_tmp = torch.cat([sampled_neg_repres_tmp1, sampled_neg_repres_tmp2], dim=-1)  # C x N_other
                    other_sim_mat = compute_similarity(self.cfg, filtered_repres, sampled_neg_repres_tmp.T, sim_type)  # N_filtered x N_other
                    other_sim_mat /= tau
                    sim_matrix_comb1 = torch.cat([sim_matrix, other_sim_mat], dim=-1)    # N_filtered x (N_filtered + N_other)
                    sim_matrix_comb2 = torch.cat([sim_matrix.T, other_sim_mat], dim=-1)  # N_filtered x (N_filtered + N_other)
                    # augment GT
                    N_filtered, N_other = other_sim_mat.shape
                    groundtruth = torch.cat([groundtruth, torch.zeros(N_filtered, N_other).cuda()], dim=-1)  # N_filtered x (N_filtered + N_other)
                    log_prob1 = nn.functional.log_softmax(sim_matrix_comb1, dim=1)
                    align_loss_tmp1 = loss_func(log_prob1, groundtruth)
                    log_prob2 = nn.functional.log_softmax(sim_matrix_comb2, dim=1)
                    align_loss_tmp2 = loss_func(log_prob2, groundtruth)
                    align_loss = (align_loss_tmp1 + align_loss_tmp2) / 2
            else:
                align_loss = torch.tensor(0.).cuda()
        else:
            raise NotImplementedError

        return align_loss

    def set_cost_eval(self, eval_cost=False):
        # need to set for each time eval
        self.eval_cost = eval_cost  # True or False
        self.cost_dict = {'IT1': 0, 'IT2': 0}  # cost: time & params
        self.time_meter1 = AverageMeter()
        self.time_meter2 = AverageMeter()

    def get_cost_eval(self):
        self.cost_dict['IT1'] = self.time_meter1.avg
        self.cost_dict['IT2'] = self.time_meter2.avg
        return self.cost_dict.copy()

    def forward(self, supports, queries, support_kps_, support_kp_mask_, **kwargs):
        '''
        input arguments
        supports (image): S x B1 x 3 x H x W (S episodes' support images)
        queries (image):  S x B2 x 3 x H x W (S episodes' query images)
        support_kps: S x B1 x N x 2, ranges -1~1 (continuous)
        support_kp_mask: S x B1 x N

        output arguments
        heatmaps_predict_list: a list of heatmaps, each is S x B2 x N x H x W

        '''
        S, B1, C, H, W = supports.shape  # S episodes, B1 support images
        _, B2, _, _, _ = queries.shape   # S episodes, B2 query images
        B_total = B1 + B2

        # supports = supports.reshape(S*B1, C, H, W)
        # queries = queries.reshape(S*B2, C, H, W)
        # support_features_list = self.encoder(x=supports)  # A list of features, each is (S*B1) x C x H x W
        # query_features_list = self.encoder(x=queries)     # A list of features, each is (S*B2) x C x H x W


        in_ims = torch.cat([supports, queries], dim=1)  # S x (B1+B2) x C x H x W
        in_ims = in_ims.reshape(S * (B1 + B2), C, H, W) # (S * (B1+B2)) x C x H x W

        if self.eval_cost:
            time_start = time.time()

        if self.trunk == 'RESNET50':
            out_features = self.encoder(x=in_ims)  # [im_num x C x h x w]
        elif self.trunk == 'DINO':
            dino_patch_size = self.cfg.MODEL.ENCODER[self.trunk].PATCH_SIZE
            dino_h = H // dino_patch_size
            out_features, dino_attn = self.encoder(x=in_ims)  # im_num x (1+h*w) x C, im_num x N_head x (1+h*w) x (1+h*w)
            out_features = out_features[:, 1:]  # im_num x (h*w) x C
            dino_feature_channel = self.encoder.num_features
            out_features = out_features.permute(0, 2, 1).reshape(S * B_total, dino_feature_channel, dino_h, dino_h)  # im_num x C x h x w
            # cls_attn = (dino_attn[:, :, 0, 1:]).mean(1).detach()  # B x Np (equal h x w)
            # cls_attn = cls_attn.reshape(S*B_total, dino_h, dino_h).cpu().numpy()  # B x h x w

            # simply upsample to get multi-scale features
            out_features = [nn.functional.adaptive_avg_pool2d(out_features, output_size=H // ds) \
                            for ds in self.downsize_factor_list]
            # if len(self.downsize_factor_list) >= 2:
            #     out_features = [nn.functional.adaptive_avg_pool2d(out_features, output_size=H // ds) \
            #         for ds in self.downsize_factor_list]
            # else:
            #     out_features = [out_features]
        else:
            raise NotImplementedError


        if self.eval_cost:
            self.time_meter1.update((time.time() - time_start) / in_ims.shape[0], n=in_ims.shape[0])  # sec / im
            time_start = time.time()

        context_mode = 'soft_fiber_gaussian'  # 'soft_fiber_bilinear'

        downsize_factor_list = self.downsize_factor_list
        image_length    = self.cfg.DATASET.SQUARE_IMAGE_LENGTH
        num_scales = len(downsize_factor_list)
        heatmaps_list = []

        # Loss for domain alignment
        alignment_type = self.cfg.LOSS.DOMAIN_ALIGNMENT.TYPE
        if (alignment_type != None) and (kwargs.get('align_kps') is not None):
            use_proto = self.cfg.LOSS.DOMAIN_ALIGNMENT.USE_PROTO
            sim_type = self.cfg.LOSS.DOMAIN_ALIGNMENT.SIMILARITY.TYPE
            tau      = self.cfg.LOSS.DOMAIN_ALIGNMENT.SIMILARITY.TAU
            align_kps = kwargs['align_kps']  # S x (B1+B2) x N_align x 2
            align_kps_mask = kwargs['align_kps_mask']  # S x (B1+B2) x N_align

            out_features_last = out_features[-1]  # (S * (B1+B2)) x C x H x W
            N_align = align_kps.shape[2]  # number of aligned keypoints per image
            align_kps = align_kps.reshape(S * B_total, N_align, 2)        # (S*(B1+B2)) x N_align x 2
            align_kps_mask = align_kps_mask.reshape(S * B_total, N_align) # (S*(B1+B2)) x N_align
            sigma = self.cfg.DATASET.SIGMA  # (image space)
            downsize_factor = downsize_factor_list[-1]
            # (S * (B1+B2)) x C x N_align
            align_repres, _ = extract_representations(out_features_last, align_kps, align_kps_mask, context_mode=context_mode,
                            sigma=sigma, downsize_factor=downsize_factor, image_length=image_length, together_trans_features=None)

            if use_proto:  # organize samples by episodes
                C_f = align_repres.shape[1]
                align_kps_mask = align_kps_mask.reshape(S, B_total, N_align).sum(1)  # S x N_align
                mask_tmp = align_kps_mask > 0
                align_kps_mask = align_kps_mask * mask_tmp + (~mask_tmp) * B_total  # avoid divide 0
                align_repres = align_repres.reshape(S, B_total, C_f, N_align).sum(1)  # S x C x N_align
                align_repres /= align_kps_mask.unsqueeze(1)
                align_kps_mask = mask_tmp.long()  # S x N_align, set 1 or 0 for keypoint mask

            # ---------------------------------------------------------------------------------------------
            # extract sampled negative kps representations
            use_sampled_neg = True if self.cfg.LOSS.DOMAIN_ALIGNMENT.SAMPLED_NEG.TYPE is not None else False
            if (use_sampled_neg == True) and (kwargs.get('sampled_neg_kps') is not None):
                neg_kps = kwargs['sampled_neg_kps']  # S x (B1+B2) x N_sampled x 2
                neg_kps = neg_kps.flatten(0, 1)      # (S*(B1+B2)) x N_sampled x 2
                neg_kps_mask = torch.ones(neg_kps.shape[:2]).cuda()  # (S*(B1+B2)) x N_sampled
                # (S * (B1+B2)) x C x N_sampled
                neg_repres, _ = extract_representations(out_features_last, neg_kps, neg_kps_mask, context_mode=context_mode,
                                sigma=sigma, downsize_factor=downsize_factor, image_length=image_length, together_trans_features=None)
                if use_proto:  # organize samples by episodes
                    _, C_f, N_sampled = neg_repres.shape
                    # S x C x ((B1+B2)*N_sample) (use_proto=True)
                    neg_repres = neg_repres.reshape(S, B_total, C_f, N_sampled).permute(0, 2, 1, 3).reshape(S, C_f, B_total*N_sampled)
            else:
                neg_repres = None
            # ---------------------------------------------------------------------------------------------

            align_loss = self.domain_alignment(align_repres, align_kps_mask, alignment_type, sim_type, tau, \
                                               sampled_neg_repres=neg_repres)
        else:
            align_loss = None

        # Predict corresponding keypoints in query image
        for i in range(num_scales):
            heatmaps_list_per_scale = []
            for epi_ind in range(S):
                support_features = out_features[i][epi_ind*B_total: epi_ind*B_total+B1]       # B1 x C x H x W
                query_features   = out_features[i][epi_ind*B_total+B1:(epi_ind + 1)*B_total]  # B2 x C x H x W
                support_kps = support_kps_[epi_ind]          # B1 x N x 2, ranges -1~1 (continuous)
                support_kp_mask = support_kp_mask_[epi_ind]  # B1 x N

                sigma = self.cfg.DATASET.SIGMA  # (image space)
                downsize_factor = downsize_factor_list[i]
                # B1 x C x N
                support_repres, conv_query_features = extract_representations(support_features, support_kps, support_kp_mask, context_mode=context_mode,
                                sigma=sigma, downsize_factor=downsize_factor, image_length=image_length, together_trans_features=None)


                '''Either normalize + l2 loss (mse) or sigmoid + bec loss is better for WG based FSL, according to my 
                experiments. Using normalization is like to do cosine-similarity scores, and this operation has good 
                generalization to novel categories only when there is 1 linear layer. When the classifier has multiple layers, 
                it doesn't have better performance and even have drops.
                '''
                # Weight Generator (WG)
                base_weights = self.base_weights
                if self.normalize_features:
                    support_repres = nn.functional.normalize(support_repres, p=2, dim=1, eps=1e-12)
                    if base_weights is not None:
                        base_weights = nn.functional.normalize(base_weights, p=2, dim=1, eps=1e-12)
                if self.wg_type == 'favg':
                    avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                    weights = avg_support_repres.transpose(1, 0)
                    weights_list, bias_list = [weights], [None]
                elif self.wg_type == 'favg-diag' or self.wg_type == 'favg-mat' or self.wg_type == 'favg-mat2':
                    avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                    weights = self.weight_generator_list[i](avg_support_repres.transpose(1, 0))  # N x C
                    weights_list, bias_list = [weights], [None]
                elif self.wg_type == 'favg-conv2' or self.wg_type == 'favg-conv2-ablation':  # supports to generate multiple filters
                    avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                    # weights_list: a list with each N x C x w x w; out_bias: None or N
                    weights_list, bias_list = self.weight_generator_list[i](avg_support_repres.transpose(1, 0))
                    # weights = weights_list[0]  # N x C x w x w
                elif self.wg_type == 'favg-diag-att' or self.wg_type == 'favg-mat-att':
                    # base_weights: N_{base} x C
                    # support_repres: B1 x C x N
                    weights = self.weight_generator_list[i](base_weights, support_repres.transpose(2, 1), support_kp_mask)  # N x C
                    weights_list, bias_list = [weights], [None]
                else:
                    raise NotImplementedError

                if self.cfg.MODEL.LWMODEL.QUERY_SUPER_RESO is not None:
                    query_features = self.query_sr_list[i](query_features)

                for j, per_weights in enumerate(weights_list):
                    # the parameters of localization_net is injected by ``weights''
                    scaled_conv_factor = self.cfg.MODEL.LWMODEL.KD_SCALED_CONV_FACTOR
                    per_bias = bias_list[j]
                    heatmaps_predict = self.ln(query_features, per_weights, per_bias, \
                                               scaled_conv_factor=scaled_conv_factor)  # B2 x N x H x W

                    # post-process for heatmaps_predict (different loss function has different requirements)
                    if self.cfg.LOSS.TYPE == 'MSE':
                        if self.cfg.LOSS.MSE.CLAMP_HEATMAP:  # 'normalize-mse'
                            heatmaps_predict = torch.clamp(heatmaps_predict, 0, 1)
                    elif self.cfg.LOSS.TYPE == 'sigmoid-bce':
                        heatmaps_predict = torch.sigmoid(heatmaps_predict)
                    elif self.cfg.LOSS.TYPE == 'GM_GM_L2':
                        heatmaps_predict = heatmap_normalize_layer(heatmaps_predict, self.cfg.LOSS.GM_GM_L2.PRED_PROB_NORM)
                    else:
                        pass

                    heatmaps_list_per_scale.append(heatmaps_predict)
            # heatmaps_list_per_scale has S * G heatmaps, with each of B2 x N x H x W (S episodes & G groups)
            num_groups = len(weights_list)
            for g in range(num_groups):  # G groups
                heatmaps_per_group = torch.stack(heatmaps_list_per_scale[g::num_groups], dim=0)  # S x B2 x N x H x W
                heatmaps_list.append(heatmaps_per_group)  # has G heatmaps, with each of S x B2 x N x H x W

        # heatmaps_list_copy_for_draw = copy.deepcopy(heatmaps_list)
        if self.cfg.MODEL.LWMODEL.KD_SUPER_RESO.TYPE is not None:
            heatmaps_list = self.heatmap_upscale(heatmaps_list)

        if self.eval_cost:
            num_measured_kps = torch.prod(torch.tensor(heatmaps_list[0].shape[:3])).numpy()  # S * B2 * N
            self.time_meter2.update((time.time() - time_start) / num_measured_kps, n=num_measured_kps)  # sec / kp

        # if self.heatmap_to_coord_type is not None:
        #     coords_list = self.heatmap_to_coord(heatmaps_list)
        #     return coords_list  # a list whose each element has size of B x N x 2 ((x, y) in range -1~1)

        # heatmaps_list: a list whose each element has size of S x B x N x H x W
        # align_loss: 0 or a scalar
        # return (heatmaps_list, align_loss, heatmaps_list_copy_for_draw)
        return (heatmaps_list, align_loss)

def get_lw_base_model(cfg, **kwargs):
    lw_base_model = LWModelBase(cfg)
    return lw_base_model

def get_lw_model(cfg, **kwargs):
    lw_model = LWModel(cfg)
    lw_model.init_weights(base_model_file=cfg.MODEL.BASE_MODEL_FILE, \
                          pretrained=cfg.MODEL.PRETRAINED)
    return lw_model

def get_lw_model_ms(cfg, **kwargs):
    lw_model = LWModelMultiScale(cfg)
    lw_model.init_weights(base_model_file=cfg.MODEL.BASE_MODEL_FILE, \
                          pretrained=cfg.MODEL.PRETRAINED)
    return lw_model












