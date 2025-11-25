# -*- coding: utf-8 -*-
import copy
import os
import torch
from torch import nn
from torchvision.models.resnet import resnet34, resnet50
import numpy as np
from network.models_gridms2 import extract_representations, average_representations2
import network.clip_kd as clip
import network.clip_kd.model as transformer_lib
import network.blip_kd.blip_for_kd as blip
import logging
from utils.utils import SoftArgmax, compute_similarity, compute_similarity2, compute_distance, AverageMeter, apply_noise
import time
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class AdaptationNet(nn.Module):
    def __init__(self, adapt_type='RESIDUAL_REFINE', block_type='transformer', dim=768, heads=12, blocks=1, dim_out=512):
        '''
        :param type:
        :param dim:
        :param out_dim:
        :param layers:
        '''
        super(AdaptationNet, self).__init__()
        self.adapt_type = adapt_type
        self.block_type = block_type
        self.dim = dim
        self.heads = heads
        self.blocks = blocks
        self.dim_out = dim_out
        if (self.adapt_type=='RESIDUAL_REFINE') or (self.adapt_type=='RESIDUAL_REFINE2'):
            if self.block_type == 'transformer':
                self.net = transformer_lib.Transformer(width=dim, layers=blocks, heads=heads, attn_mask=None)
                self.ln = transformer_lib.LayerNorm(dim)
                self.proj = nn.Parameter(torch.empty(dim, dim_out))
                # self.ln2 = transformer_lib.LayerNorm(dim_out)
                # self.proj2 = nn.Parameter(torch.empty(dim_out, dim_out))

                nn.init.normal_(self.proj, mean=0, std=dim ** -0.5)  # TODO: Important to init proj weights to keep neuron value reasonable
                proj_std = (self.net.width ** -0.5) * ((2 * self.net.layers) ** -0.5)
                attn_std = self.net.width ** -0.5
                fc_std = (2 * self.net.width) ** -0.5
                for block in self.net.resblocks:
                    nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                    nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                    nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                    nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            elif self.block_type == 'bottleneck':  # 'bottleneck' does not need 'heads' as it is a Conv block
                net = []
                assert (dim // 4) * 4 == dim, 'Bottleneck has channel expansion rate of 4'
                for i in range(blocks):
                    net.append(transformer_lib.Bottleneck(inplanes=dim, planes=dim//4, stride=1))
                self.net = nn.Sequential(*net)
                self.bn = torch.nn.BatchNorm2d(dim)
                self.proj = nn.Conv2d(dim, dim_out, kernel_size=1, stride=1, padding=0)
                nn.init.normal_(self.proj.weight, mean=0.0, std=dim ** -0.5)
                nn.init.constant_(self.proj.bias, val=0.0)
            elif self.block_type == None:  # no params.
                pass
            else:
                raise NotImplementedError
        # elif self.adapt_type==''
        else:
            raise NotImplementedError

    def forward(self, x, z=None, **kwargs):
        '''
        for block_type == 'transformer':
        :param x: primary feature with size of N*L*D (N is batch size, L is senquence length, D is dim)

        for self.block_type == 'bottleneck':
        :param x: primary feature with size of B*L*D (1+H*W)--> B x C x H x W

        :param z: secondary feature with size of N*L2*D
        :param kwargs:
        :return:
        '''
        if (self.adapt_type == 'RESIDUAL_REFINE') or (self.adapt_type == 'RESIDUAL_REFINE2'):
            if self.block_type == 'transformer':
                x = x.permute(1, 0, 2)  # NLD -> LND
                z = z.permute(1, 0, 2)  # NLD -> LND
                z = self.net(z)  # LND
                z = self.ln(z) @ self.proj
                x = x + z  # We can further add ln+proj here.
                x = x.permute(1, 0, 2)  # LND -> NLD
            elif self.block_type == 'bottleneck':
                h = kwargs['h']
                w = kwargs['w']
                n, l, d = x.shape
                d2 = z.shape[-1]
                # assert (1+h*w) == l, 'the number of image tokens should be right.'
                cls_token = x[:, 0, :]
                x = (x[:, 1:, :]).permute(0, 2, 1).reshape(n, d, h, w)
                z = (z[:, 1:, :]).permute(0, 2, 1).reshape(n, d2, h, w)
                z = self.net(z)
                z = self.proj(self.bn(z))
                x = x + z
                x = x.reshape(n, d, h*w).permute(0, 2, 1)
                x = torch.cat([cls_token.unsqueeze(1), x], dim=1)   # NLD
            elif self.block_type == None:
                pass
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return x

# ClassAgnostic convert single attentive feature into a heatmap
class LocalizationCNN(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, num_block=1):
        super(LocalizationCNN, self).__init__()
        nets = []
        nets.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=3,stride=1,padding=1))
        nets.append(nn.ReLU())
        assert num_block >= 1, 'block number should >=1.'
        for i in range(num_block-1):
            conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1,padding=1)
            relu = nn.ReLU()
            nets.append(conv)
            nets.append(relu)
        nets.append(nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1,padding=0))
        self.net = nn.Sequential(*nets)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
    #         # nn.init.trunc_normal_(m.weight, mean=0., std=.005, a=-0.02, b=0.02)
    #         nn.init.normal_(m.weight, mean=0, std=0.001)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        '''
        x: N x C (in_channels) x H x W
        return: N x 1 x H x W
        '''
        out = self.net(x)
        return out

class LocalizationTransformer(nn.Module):
    def __init__(self, in_channels=2048, hidden_channels=512, num_block=1):
        super(LocalizationTransformer, self).__init__()

        self.ln = transformer_lib.LayerNorm(in_channels)
        self.proj = nn.Parameter(torch.empty(in_channels, hidden_channels))
        nn.init.normal_(self.proj, mean=0, std=in_channels ** -0.5)

        heads = hidden_channels//64
        assert heads*64 == hidden_channels, 'hidden_channels should be divisible by 64.'
        self.net = transformer_lib.Transformer(width=hidden_channels, layers=num_block, heads=heads, attn_mask=None)
        self.ln_post = transformer_lib.LayerNorm(hidden_channels)
        self.proj_post = nn.Parameter(torch.empty(hidden_channels, 1))
        proj_std = (self.net.width ** -0.5) * ((2 * self.net.layers) ** -0.5)
        attn_std = self.net.width ** -0.5
        fc_std = (2 * self.net.width) ** -0.5
        for block in self.net.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.proj_post, mean=0, std=((2*hidden_channels) ** -0.5) * 0.1)

    def forward(self, x: torch.Tensor):
        '''
        x: N x C (in_channels) x H x W
        return: N x 1 x H x W
        '''
        N, C, H, W = x.shape
        x = x.reshape(N, C, H*W).permute(2, 0, 1)  # (H*W) x N x C (LND)
        x = self.ln(x) @ self.proj
        x = self.net(x)
        x = self.ln_post(x) @ self.proj_post  # (H*W) x N x 1
        x = x.permute(1, 2, 0).reshape(N, 1, H, W)
        return x

class CrossAttTransformer(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=1024, num_block=1):
        super(CrossAttTransformer, self).__init__()
        assert in_channels == hidden_channels, 'we assume we keep dim unchanged.'

        heads = hidden_channels // 64
        assert heads * 64 == hidden_channels, 'hidden_channels should be divisible by 64.'
        self.num_block = num_block
        self.cross_att = nn.ModuleList()
        for i in range(num_block):
            self.cross_att.append(transformer_lib.CrossAttentionBlock(hidden_channels, heads))
        # if num_block >= 2:  # at least 1 SA transformer
        #     self.transformer = transformer_lib.Transformer(hidden_channels, layers=num_block-1, heads=heads)
        # else:
        #     self.transformer = None

        self.ln_post = transformer_lib.LayerNorm(2*hidden_channels)  # because we concat feature maps, so dim is 2C
        self.proj_post = nn.Parameter(torch.empty(2*hidden_channels, 1))

        proj_std = (hidden_channels ** -0.5) * ((2 * num_block) ** -0.5)
        attn_std = hidden_channels ** -0.5
        fc_std = (2 * hidden_channels) ** -0.5

        nn.init.normal_(self.proj_post, mean=0, std=((2*hidden_channels) ** -0.5) * 0.1)

        for block in self.cross_att:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if num_block >= 2:  # at least 1 SA transformer
        #     for block in self.transformer.resblocks:
        #         nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #         nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        '''
        q:  N x C (in_channels)
        kv: B x C (in_channels) x H x W
        return: N x 1 x H x W
        '''
        B, C, H, W = kv.shape
        N = q.shape[0]
        q_tmp = q.unsqueeze(1).expand(N, B, C)  # N x B x C
        kv_tmp = kv.reshape(B, C, H*W).permute(2, 0, 1)  # (H*W) x B x C (LND)
        for i in range(self.num_block):
            q_tmp = self.cross_att[i](q_tmp, kv_tmp)  # N x B x C
        enhanced_q = q_tmp
        # if self.num_block >= 2:  # at least 1 SA transformer
        #     enhanced_q = self.transformer(enhanced_q)
        enhanced_q = enhanced_q.permute(1, 0, 2).unsqueeze(-1).unsqueeze(-1)  # B x N x C x 1 x 1
        enhanced_q = enhanced_q.expand(B, N, C, H, W)  # B x N x C x H x W
        kv_expand = kv.unsqueeze(1).expand(B, N, C, H, W)  # B x N x C x H x W
        enhanced_out = torch.cat([enhanced_q, kv_expand], dim=2)  # B x N x (2C) x H x W
        enhanced_out = enhanced_out.reshape(B, N, 2*C, H*W).permute(0, 1, 3, 2)  # B x N x (H*W) x (2C)
        enhanced_out = self.ln_post(enhanced_out) @ self.proj_post  # B x N x (H*W) x 1
        enhanced_out = enhanced_out.reshape(B, N, H, W)
        return enhanced_out

def progressive_sampling(mask, topk):
    '''
    Sample a number from valid entries progressively. We firstly sample from top1; if failed, top2, etc.
    :param mask: N_repeat x K_texts
    :param topk: topk <= K_texts
    :return: success_flag, sampled_n, sampled_k
    '''
    mask_per_k_dim = mask.sum(0)
    success_flag = False
    sampled_n = 0
    sampled_k = 0
    for i in range(topk):
        if mask_per_k_dim[i] > 0:
            valid_inds = torch.nonzero(mask[:, i])
            ind = torch.randint(len(valid_inds), [1])  # sample 1 number
            ind = ind[0]
            sampled_n = valid_inds[ind]
            sampled_k = i
            success_flag = True
            break
    return success_flag, sampled_n, sampled_k

def even_sampling(mask, topk):
    '''
    Sample a number from valid entries evenly within topk
    :param mask: N_repeat x K_texts
    :param topk: topk <= K_texts
    :return: success_flag, sampled_n, sampled_k
    '''
    mask_per_k_dim = mask.sum(0)
    if sum(mask_per_k_dim[:topk]) <= 0:
        return False, 0, 0
    mask_tmp = mask[:, :topk]  # N_repeat x topk
    valid_inds = torch.nonzero(mask_tmp)
    ind = torch.randint(len(valid_inds), [1])  # sample 1 number
    ind = ind[0]
    sampled_n, sampled_k = valid_inds[ind]
    return True, sampled_n, sampled_k

class OpenKDModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(OpenKDModel, self).__init__()
        self.cfg = cfg
        # encoder related parameters
        trunk = cfg.MODEL.ENCODER.TRUNK
        self.trunk = trunk
        if self.trunk == 'CLIP':
            clip_name = cfg.MODEL.ENCODER.CLIP.NAME
            clip_weights_root = cfg.MODEL.ENCODER.CLIP.WEIGHTS_ROOT
            clip_weights_path = os.path.join(clip_weights_root, clip_name)
            device = 'cpu'  # 'cuda' (half floating precision) or 'cpu' (floating precision)
            # def _convert_image_to_rgb(image):
            #     return image.convert("RGB")
            #
            # def _transform(n_px):
            #     return Compose([
            #         Resize(n_px, interpolation=BICUBIC),
            #         CenterCrop(n_px),
            #         _convert_image_to_rgb,
            #         ToTensor(),
            #         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            #     ])
            vision_layer_to_tune = cfg.MODEL.ENCODER.CLIP.VISION_LAYER_TO_TUNE  # -1, disable (all tune); 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
            text_layer_to_tune = cfg.MODEL.ENCODER.CLIP.TEXT_LAYER_TO_TUNE  # -1, disable (all tune); 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
            input_resolution = cfg.DATASET.SQUARE_IMAGE_LENGTH
            self.encoder, preprocess = clip.load(clip_weights_path, device=device,
                                                 vision_layer_to_tune=vision_layer_to_tune,
                                                 text_layer_to_tune=text_layer_to_tune,
                                                 new_im_reso=input_resolution)
            if cfg.MODEL.ENCODER.CLIP.START_FROM_TRAIN:
                self.encoder = self.encoder.train()  # after clip.load(), the model is in eval status.

            input_resolution = self.encoder.visual.input_resolution
            context_length = self.encoder.context_length
            vocab_size = self.encoder.vocab_size

            print('==>CLIP {} loaded with information as:'.format(clip_name))
            print("CLIP parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.encoder.parameters()]):,}")
            print("Input resolution:", input_resolution)
            print("Context length:", context_length)
            print("Vocab size:", vocab_size)
            # print(preprocess)

            # ------------------------------------------------------
            # Some parameters required by subsequent adaptation net
            self.is_vit = isinstance(self.encoder.visual, transformer_lib.VisionTransformer)
            if self.is_vit:
                visual_dim = self.encoder.visual.positional_embedding.shape[1]
                visual_dim_out = self.encoder.visual.output_dim
                visual_heads = self.encoder.visual.heads
            else:  # Visual encoder is transformer_lib.ModifiedResNet
                visual_dim_out, visual_dim = self.encoder.visual.attnpool.c_proj.weight.shape
                visual_heads = self.encoder.visual.heads  # 'bottleneck' does not need 'heads' as it is a Conv block
                # assert visual_heads * 64 == visual_dim, 'The dim should be multiple of 64.'
            textual_dim, textual_dim_out = self.encoder.text_projection.shape  # projection is nn.Parameters
            textual_heads = self.encoder.transformer.heads

            # modify input resolution & Gaussian STD
            # cfg.DATASET.SQUARE_IMAGE_LENGTH = input_resolution
            cfg.DATASET.SIGMA = input_resolution/384 * 14
        elif self.trunk == 'BLIP':
            blip_name = cfg.MODEL.ENCODER.BLIP.NAME
            blip_vit_type = blip_name.replace('.', '_').split('_')[1]  # 'base' or 'large'
            blip_weights_root = cfg.MODEL.ENCODER.BLIP.WEIGHTS_ROOT
            blip_weights_path = os.path.join(blip_weights_root, blip_name)

            vision_layer_to_tune = cfg.MODEL.ENCODER.BLIP.VISION_LAYER_TO_TUNE  # -1, disable (all tune); 0, no layer to tune (all freeze); 1, proj to tune
            text_layer_to_tune = cfg.MODEL.ENCODER.BLIP.TEXT_LAYER_TO_TUNE  # -1, disable (all tune); 0, no layer to tune (all freeze); 1, proj to tune

            self.encoder = blip.load(
                pretrained=blip_weights_path,
                image_size=cfg.DATASET.SQUARE_IMAGE_LENGTH,
                vit=blip_vit_type,
                vision_layer_to_tune=vision_layer_to_tune,
                text_layer_to_tune=text_layer_to_tune
            )

            print('==>BLIP {} loaded with information as:'.format(blip_name))
            print("BLIP parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.encoder.parameters()]):,}")

            # ------------------------------------------------------
            # blip's default patch size is 16 x 16, input image reso is 224, patch grids 14 x 14=196
            # visual_dim = textual_dim = 768, projected dim is 256, namely 768-->256
            visual_dim_out, visual_dim = self.encoder.vision_proj.weight.shape
            visual_heads = visual_dim_out // 64
            textual_dim_out, textual_dim = self.encoder.text_proj.weight.shape
            textual_heads = textual_dim_out // 64
            assert (visual_heads*64 == visual_dim_out) and (textual_heads*64 == textual_dim_out), 'BLIP feature dim should be divisible by 64'

            # modify input resolution & Gaussian STD
            cfg.DATASET.SIGMA = self.encoder.image_size / 384 * 14
        else:
            raise NotImplementedError

        self.adaptation_net_type = cfg.MODEL.ADAPTATION_NET.TYPE
        if self.adaptation_net_type in [None, 'RESIDUAL_REFINE']:
            x_visual_dim = visual_dim_out
            z_visual_dim = visual_dim
            z_visual_heads = visual_heads
            x_textual_dim = textual_dim_out
            z_textual_dim = textual_dim
            z_textual_heads = textual_heads
        elif self.adaptation_net_type == 'RESIDUAL_REFINE2':
            x_visual_dim = visual_dim_out
            z_visual_dim = visual_dim_out
            z_visual_heads = visual_dim_out // 64
            x_textual_dim = textual_dim_out
            z_textual_dim = textual_dim_out
            z_textual_heads = textual_dim_out // 64
            assert z_visual_heads * 64 == z_visual_dim, 'dim should be multiple of 64 (each head)'
            assert z_textual_heads * 64 == z_textual_dim, 'dim should be multiple of 64 (each head)'
        else:
            raise NotImplementedError

        if self.adaptation_net_type == None:  # no adaptation
            pass
        elif self.adaptation_net_type in ['RESIDUAL_REFINE', 'RESIDUAL_REFINE2']:
            self.vision_anet = AdaptationNet(
                adapt_type=self.adaptation_net_type,
                block_type=cfg.MODEL.ADAPTATION_NET.VISION_ANET.TYPE,
                dim=z_visual_dim,
                heads=z_visual_heads,
                blocks=cfg.MODEL.ADAPTATION_NET.VISION_ANET.NUM_BLOCKS,
                dim_out=x_visual_dim
            )
            self.text_anet = AdaptationNet(
                adapt_type=self.adaptation_net_type,
                block_type=cfg.MODEL.ADAPTATION_NET.TEXT_ANET.TYPE,
                dim=z_textual_dim,
                heads=z_textual_heads,
                blocks=cfg.MODEL.ADAPTATION_NET.TEXT_ANET.NUM_BLOCKS,
                dim_out=x_textual_dim
            )
        else:
            raise NotImplementedError

        self.feature_map_up_scale_number = cfg.MODEL.FEATURE_UP
        if (self.feature_map_up_scale_number is not None) and (self.feature_map_up_scale_number != 1):
            self.feature_sr = nn.Upsample(scale_factor=self.feature_map_up_scale_number, mode='bilinear')
        else:
            self.feature_sr = None

        self.visual_prompt_extraction_type = cfg.MODEL.VISUAL_PROMPT_EXTRACTOR.TYPE
        self.correlation_decoding_type = cfg.MODEL.CORRELATION_DECODING_NET.TYPE
        corr_decode_cfg = cfg.MODEL.CORRELATION_DECODING_NET
        if self.correlation_decoding_type == 'COSINE_SIMILARITY':
            pass
        elif self.correlation_decoding_type in ['SIMPLE_CORRELATION', 'RELATION_PAIRS']:
            if self.correlation_decoding_type == 'SIMPLE_CORRELATION':
                decoder = corr_decode_cfg.SIMPLE_CORRELATION.DECODER  # 'conv' or 'transformer'
                num_blocks = corr_decode_cfg.SIMPLE_CORRELATION.NUM_BLOCKS
                in_dim_tmp, hidden_dim_tmp = (x_visual_dim, x_visual_dim)
            elif self.correlation_decoding_type == 'RELATION_PAIRS':
                decoder = corr_decode_cfg.RELATION_PAIRS.DECODER  # 'conv' or 'transformer'
                num_blocks = corr_decode_cfg.RELATION_PAIRS.NUM_BLOCKS
                in_dim_tmp, hidden_dim_tmp = (2*x_visual_dim, x_visual_dim)
            if decoder == 'conv':
                self.correlation_decoding_net = LocalizationCNN(in_dim_tmp, hidden_dim_tmp, num_block=num_blocks)
            elif decoder == 'transformer':
                self.correlation_decoding_net = LocalizationTransformer(in_dim_tmp, hidden_dim_tmp, num_block=num_blocks)
            else:
                raise NotImplementedError
        elif self.correlation_decoding_type == 'CROSS_ATT':
            num_blocks = corr_decode_cfg.CROSS_ATT.NUM_BLOCKS
            in_dim_tmp, hidden_dim_tmp =  x_visual_dim, x_visual_dim
            self.correlation_decoding_net = CrossAttTransformer(in_dim_tmp, hidden_dim_tmp, num_block=num_blocks)
        else:
            raise NotImplementedError

        self.super_reso_type = cfg.MODEL.SUPER_RESO.TYPE
        self.super_reso_net_type = cfg.MODEL.SUPER_RESO.NET_TYPE
        self.super_reso_up_scale = cfg.MODEL.SUPER_RESO.UP_SCALE if self.super_reso_net_type is not None else 1.0
        if self.super_reso_type == 'prompt_agnostic':
            if self.super_reso_net_type is None:
                self.sr_up = nn.Identity()
            elif self.super_reso_net_type == 'bilinear':
                self.sr_up = nn.Upsample(scale_factor=self.super_reso_up_scale, mode='bilinear') \
                    if self.super_reso_up_scale > 1.0 else nn.Identity()
            elif self.super_reso_net_type == 'conv1':
                self.sr_up = nn.Sequential(
                    nn.Upsample(scale_factor=self.super_reso_up_scale, mode='bilinear'),
                    nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
                )
            elif self.super_reso_net_type == 'conv2':
                self.sr_up = nn.Sequential(
                    nn.Upsample(scale_factor=self.super_reso_up_scale, mode='bilinear'),
                    nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),  # single-channel 5x5 convolution
                    nn.ReLU(),
                    nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
                )
                # up_sr[1].weight.data.normal_(mean=1 / 5 ** 2, std=0.001)
                # up_sr[1].bias.data.zero_()
                # up_sr[3].weight.data.normal_(mean=1 / 5 ** 2, std=0.001)
                # up_sr[3].bias.data.zero_()
            else:
                raise NotImplementedError
        elif self.super_reso_type == 'prompt_specific':
            pass
        else:
            raise NotImplementedError

        # hyper-parameters for domain alignment
        cfg_align = self.cfg.LOSS.DOMAIN_ALIGNMENT
        self.alignment_type = cfg_align.TYPE
        self.v_t_align      = cfg_align.V_T_ALIGN
        self.v_v_align      = cfg_align.V_V_ALIGN
        self.t_t_align      = cfg_align.T_T_ALIGN
        self.use_query_kps  = cfg_align.USE_QUERY_KPS
        self.use_proto_main = cfg_align.USE_PROTO_MAIN
        self.use_proto_itpl = cfg_align.USE_PROTO_ITPL
        self.sg_textual     = cfg_align.SG_TEXTUAL
        self.sim_type       = cfg_align.SIMILARITY.TYPE
        self.tau            = cfg_align.SIMILARITY.TAU
        self.v_t_align_itpl = (cfg_align.V_T_ALIGN_ITPL == True) and \
                              (cfg.TRAIN.NUM_TRAIN_SHOT > 0) and \
                              (cfg.TRAIN.ENABLE_ITPL_VISUAL == True) and \
                              (cfg.TRAIN.TEXT_PROMPT_SETTING.NUM_TEXT > 0) and \
                              (cfg.TRAIN.TEXT_PROMPT_SETTING.ENABLE_ITPL_TEXT==True)

        # init cost evaluator
        self.set_cost_eval(eval_cost=False)  # by default disabled, but it can be opened manually
        self.set_complement_itpl_kps_texts_info()

        print("==>Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.parameters()]):,}")
        print("==>Learn parameters:", f"{np.sum([int(np.prod(p.shape)) if p.requires_grad else 0 for p in self.parameters()]):,}")

    def init_weights(self, pretrained=''):
        # load pretrained model
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_state_dict)
            print('==> loading pretrained model {}.'.format(pretrained))

    def set_cost_eval(self, eval_cost=False):
        # need to set for each time eval
        self.eval_cost = eval_cost  # True or False
        self.cost_dict = {'IT1': 0, 'IT2': 0, 'IT3': 0}  # cost: time & params
        self.time_meter1 = AverageMeter()  # avg CLIP processing time per episode
        self.time_meter2 = AverageMeter()  # avg adaptation time per episode
        self.time_meter3 = AverageMeter()  # avg prototype set building+correlation+detection time per episode

    def get_cost_eval(self):
        self.cost_dict['IT1'] = self.time_meter1.avg
        self.cost_dict['IT2'] = self.time_meter2.avg
        self.cost_dict['IT3'] = self.time_meter3.avg
        return self.cost_dict.copy()

    def dict2list(self, prompt_set: dict):
        prompts_combined = []
        for type in ['obj', 'text', 'image']:
            if len(prompt_set[type]) <= 0:
                continue
            prompt = prompt_set[type]  # N x C
            prompts_combined.append(prompt)
        return prompts_combined

    def prompts_combine(self, prompt_set: dict):
        prompts_combined = []
        for type in ['obj', 'text', 'image']:
            if len(prompt_set[type]) <= 0:
                continue
            prompt = prompt_set[type]  # N x C
            prompts_combined.append(prompt)
        prompts_combined = torch.cat(prompts_combined, dim=0)
        return prompts_combined

    def heatmaps_separation(self, heatmaps: torch.Tensor, prompt_set: dict):
        '''
        Separate heatmap tensor into repsective groups.
        :param heatmaps: B x N x H x W, where N is the total number of prompts
        :param prompt_set:
        :return:
        '''
        N = heatmaps.shape[1]
        cnt = 0
        heatmaps_set = {'obj': [], 'text': [], 'image': []}
        for type in ['obj', 'text', 'image']:
            n = len(prompt_set[type])
            if n <= 0:
                continue
            heatmaps_set[type] = heatmaps[:, cnt:cnt+n]
            cnt += n
        assert cnt == N, 'cnt should match N.'
        return heatmaps_set

    def visual_prompt_extraction(self, type, support_features, support_kps, support_kp_mask=None, image_width=None):
        # support_features: B1 x C x H x W
        # support_kps are in range -1~1, B1 x N x 2
        # support_kp_mask: B1 x N
        # TODO: Note we bring in human bias when extracting keypoint representations here
        if type == 'weighted_sum':  # weighted sum between Gaussian heatmap and support feature map
            context_mode = 'soft_fiber_gaussian'  # 'soft_fiber_bilinear'
            sigma = self.cfg.DATASET.SIGMA  # (image space)
            feat_width = support_features.shape[-1]
            downsize_factor = image_width // feat_width
            # B1 x C x N
            support_repres, _ = extract_representations(support_features, support_kps, support_kp_mask, context_mode=context_mode,
                            sigma=sigma, downsize_factor=downsize_factor, image_length=image_width, together_trans_features=None)
        elif type == 'feat_interpolate':  # bi-linear interpolate from feature map
            context_mode = 'soft_fiber_bilinear'
            # B1 x C x N
            support_repres, _ = extract_representations(support_features, support_kps, kp_mask=None, context_mode=context_mode)
        elif type == 'PE_interpolate': # interpolate from PE + cross attention
            pass
        else:
            raise NotImplementedError
        return support_repres

    def correlation_decoding(self, prompt_set: dict, query_features: torch.Tensor):
        # heatmaps_set = {'obj': [], 'text': [], 'image': []}
        B2, C, H, W = query_features.shape  # B2 x C x H x W
        prompt = self.prompts_combine(prompt_set)  # N x C
        N = prompt.shape[0]
        if self.correlation_decoding_type == 'COSINE_SIMILARITY':
            prompt_normalize = torch.nn.functional.normalize(prompt, p=2, dim=1)  # N x C
            prompt_normalize = prompt_normalize.reshape(1, N, C, 1, 1).expand(B2, N, C, H, W)
            query_normalize = torch.nn.functional.normalize(query_features, p=2, dim=1)  # B2 x C x H x W
            query_normalize = query_normalize.unsqueeze(1)  # B2 x 1 x C x H x W
            heatmaps = (prompt_normalize * query_normalize).sum(dim=2)  # B2 x N x H x W
        elif self.correlation_decoding_type == 'SIMPLE_CORRELATION':
            prompt_tmp = prompt.reshape(1, N, C, 1, 1).expand(B2, N, C, H, W)
            query_features_tmp = query_features.unsqueeze(1)  # B2 x 1 x C x H x W
            modulated_features = (prompt_tmp * query_features_tmp).reshape(B2*N, C, H, W)
            heatmaps = self.correlation_decoding_net(modulated_features)  # (B2*N) x 1 x H x W
            heatmaps = heatmaps.reshape(B2, N, H, W)  # B2 x N x H x W
        elif self.correlation_decoding_type == 'RELATION_PAIRS':
            prompt_tmp = prompt.reshape(1, N, C, 1, 1).expand(B2, N, C, H, W)
            query_features_tmp = query_features.unsqueeze(1).expand(B2, N, C, H, W)  # B2 x N x C x H x W
            relation_pairs = torch.cat([prompt_tmp, query_features_tmp], dim=2)  # B2 x N x (2C) x H x W
            relation_pairs = relation_pairs.reshape(B2*N, 2*C, H, W)
            heatmaps = self.correlation_decoding_net(relation_pairs)  # (B2*N) x 1 x H x W
            heatmaps = heatmaps.reshape(B2, N, H, W)  # B2 x N x H x W
        elif self.correlation_decoding_type == 'CROSS_ATT':
            heatmaps = self.correlation_decoding_net(q=prompt, kv=query_features)  # B2 x N x H x W
        else:
            raise NotImplementedError

        # heatmaps_set = self.heatmaps_separation(heatmaps, prompt_set)
        return heatmaps

    def heatmap_upscale(self, heatmaps, prompt_set):
        '''
        Upscale heatmaps, 1) obj text induced; 2) kps text induced; 3) image induced
        :param heatmaps: B2 x N x H x W
        :param prompt_set: A dict like prompt_set = {'obj': [], 'text': [], 'image': []}
        :return: heatmaps_set = {'obj': [], 'text': [], 'image': []}
        '''
        B, N, H, W = heatmaps.shape
        if self.super_reso_type == 'prompt_agnostic':
            if (self.super_reso_net_type is None) or (self.super_reso_net_type == 'bilinear'):
                heatmaps_up = self.sr_up(heatmaps)  # B x N x H2 x W2
            elif self.super_reso_net_type in ['conv1', 'conv2']:
                heatmaps = heatmaps.reshape(B*N, 1, H, W)
                heatmaps_up = self.sr_up(heatmaps)  # (B*N) x 1 x H2 x W2
            else:
                raise NotImplementedError

            H2, W2 = int(H * self.super_reso_up_scale), int(W * self.super_reso_up_scale)
            heatmaps_up = heatmaps_up.reshape(B, N, H2, W2)
            heatmaps_set = self.heatmaps_separation(heatmaps_up, prompt_set)
        elif self.super_reso_type == 'prompt_specific':
            pass
        else:
            raise NotImplementedError

        return heatmaps_set

    def openkd_heatmap_fuse(self, heatmaps_set, support_kp_mask=None, kps_texts_mask=None,
                            multi_group_supervision=True, fusing_operation='avg'):
        '''
        Fuse the obj-induced heatmaps, text-induced heatmaps, image-induced heatmaps into final output heatmaps
        :param heatmaps_set: {'obj': tensor, 'text': tensor, 'image': tensor}

        Below masks used for determining valid text/vision-induced heatmaps.
        :param kps_texts_mask: None or tensor N x T2. If None, it means all texts are valid.
        :param support_kp_mask: S x B1 x N or None. If None, it means all kps are valid.
        :return:
        '''
        # multi_group_supervision = self.cfg.LOSS.MULTI_GROUP_SUPERVISION  # True or False
        # fusing_operation = self.cfg.LOSS.OBJ_KP_HEATMAP_FUSION  # 'avg' or 'prod'
        fused_mask_sum = 0
        masks_collect = []
        heatmaps_collect = []
        if len(heatmaps_set['text']) > 0:
            text_induced_heatmaps = heatmaps_set['text']  # B2 x N x H x W
            B2, N = text_induced_heatmaps.shape[:2]
            if (kps_texts_mask is not None):
                union_kps_texts_mask = (torch.sum(kps_texts_mask, dim=1) > 0).long()  # N
            else:
                union_kps_texts_mask = torch.ones(N).cuda()  # N. Assume all texts thus heatmaps are valid
            union_kps_texts_mask = union_kps_texts_mask.reshape(1, N)  # 1 x N
            # text_induced_heatmaps *= union_kps_texts_mask.unsqueeze(-1).unsqueeze(-1)# B2 x N x H x W
            heatmaps_collect.append(text_induced_heatmaps)
            fused_mask_sum += union_kps_texts_mask  # 1 x N. Note we assume all texts are available thus heatmaps are valid
            masks_collect.append(union_kps_texts_mask)
        if (len(heatmaps_set['image']) > 0):
            image_induced_heatmaps = heatmaps_set['image']  # B2 x N x H x W
            B2, N = image_induced_heatmaps.shape[:2]
            if (support_kp_mask is not None):
                union_support_kp_mask = (torch.sum(support_kp_mask, dim=0) > 0).long()  # N
            else:
                union_support_kp_mask = torch.ones(N).cuda()  # N. Assume all kp labels thus heatmaps are valid
            union_support_kp_mask = union_support_kp_mask.reshape(1, N)  # 1 x N
            # image_induced_heatmaps *= union_support_kp_mask.unsqueeze(-1).unsqueeze(-1)# B2 x N x H x W
            heatmaps_collect.append(image_induced_heatmaps)
            fused_mask_sum += union_support_kp_mask  # B2 x N
            masks_collect.append(union_support_kp_mask)

        num_groups = len(heatmaps_collect)  # num_groups = 1 or 2
        assert num_groups > 0  # it should exist at least one of text and image induced heatmaps

        if multi_group_supervision == True:  # (each kp_heatmap * obj_heamap) then fuse
            for i in range(num_groups):
                if len(heatmaps_set['obj']) > 0:
                    obj_heatmaps = heatmaps_set['obj']  # B2 x 1 x H x W
                    if fusing_operation == 'avg':
                        heatmaps_collect[i] = (obj_heatmaps + heatmaps_collect[i]) / 2.0  # B2 x N x H x W
                    else:  # 'prod'
                        heatmaps_collect[i] = (obj_heatmaps * heatmaps_collect[i])  # B2 x N x H x W
                heatmaps_collect[i] *= masks_collect[i].unsqueeze(-1).unsqueeze(-1)  # B2 x N x H x W, masking
            heatmaps_fused = sum(heatmaps_collect) / (fused_mask_sum.unsqueeze(-1).unsqueeze(-1) + 1e-12)  # B2 x N x H x W
        else:  # False. (fused all kp_heatmap) * obj_heamap
            for i in range(num_groups):
                heatmaps_collect[i] *= masks_collect[i].unsqueeze(-1).unsqueeze(-1)  # B2 x N x H x W
            heatmaps_fused = sum(heatmaps_collect) / (fused_mask_sum.unsqueeze(-1).unsqueeze(-1) + 1e-12)  # B2 x N x H x W
            if len(heatmaps_set['obj']) > 0:
                obj_heatmaps = heatmaps_set['obj']  # B2 x 1 x H x W
                if fusing_operation == 'avg':
                    heatmaps_fused = (obj_heatmaps + heatmaps_fused) / 2.0  # B2 x N x H x W
                else:  # 'prod'
                    heatmaps_fused = (obj_heatmaps * heatmaps_fused)  # B2 x N x H x W

        # heatmaps_fused: B2 x N x H x W;
        # fused_mask_sum: 1 x N (valid heatmap has entry 1 or 2; invalid heatmap has entry 0)
        # heatmaps_collect: a list that collects valid 'text-induced heatmaps' or 'image-induced heatmaps'.
        #                   It is [tensor], or [tensor, tensor].
        # masks_collect: a list that collects kp mask. It is [tensor], or [tensor, tensor].
        return heatmaps_fused, fused_mask_sum, heatmaps_collect, masks_collect

    def feature_map_upscale(self, x):
        n, l, d = x.shape
        h = int(np.sqrt(l-1))
        w = h
        new_h = int(self.feature_map_up_scale_number * h)
        new_w = new_h
        assert (h**2+1) == l, 'token number should be (1+h*w)'
        cls_token = x[:, 0, :]  # n x d
        x = (x[:, 1:, :]).permute(0, 2, 1).reshape(n, d, h, w)
        x = self.feature_sr(x)  # n x d x (r*h) x (r*w)
        x = x.reshape(n, d, new_h * new_w).permute(0, 2, 1)
        x = torch.cat([cls_token.unsqueeze(1), x], dim=1)  # NLD
        return x

    def compute_contrastive_loss(self, repres_tmp1, mask_tmp1, repres_tmp2, mask_tmp2, sim_type, tau, loss_func,
                               use_sampled_neg, **kwargs):
        '''
        :param repres_tmp1: N x C
        :param mask_tmp1: N
        :param repres_tmp2: N x C
        :param mask_tmp2: N
        '''
        mask_combine = (mask_tmp1 * mask_tmp2).long()  # N
        if torch.sum(mask_combine) == 0:  # must exist one-one valid kp pair. thus, sim_matrix's diagonal not all zeros.
            return False, None
        mask_matrix = mask_tmp1.unsqueeze(-1) * mask_tmp2.unsqueeze(0)
        sim_matrix = compute_similarity(repres_tmp1, repres_tmp2, sim_type)  # N x N
        sim_matrix = sim_matrix - 100 * (1 - mask_matrix)  # masking
        sim_matrix /= tau  # temperature
        N = mask_combine.shape[0]
        groundtruth = torch.arange(N, dtype=torch.long).cuda()
        groundtruth = groundtruth * mask_combine - (1 - mask_combine)  # set GT labels of invalid kps to be -1

        if use_sampled_neg == False:
            align_loss_tmp1 = loss_func(sim_matrix, groundtruth)
            align_loss_tmp2 = loss_func(sim_matrix.T, groundtruth)
        else:
            raise NotImplementedError
        align_loss_tmp = (align_loss_tmp1 + align_loss_tmp2) / 2
        return True, align_loss_tmp

    def domain_alignment(self, repres1, mask1, repres2, mask2, alignment_type, sim_type, tau, match_style,
                         sampled_neg_repres=None):
        '''
        :param repres1: S x N x C
        :param mask1:   S x N
        :param repres2: S x N x C
        :param mask2:   S x N
        :param sim_type: 'cosine'
        :param tau: softmax temperature 0.05, 0.07
        :param match_style: 'within_episode' (result in S pairs), 'between_episode' (result in S(S-1)/2 pairs)
        '''
        S, N = mask1.shape

        # ---------------------------------------------------------------------------------------------
        # use sampled negative points to enhance contrastive
        use_sampled_neg = True if sampled_neg_repres is not None else False
        # ---------------------------------------------------------------------------------------------

        if alignment_type == 'align_relation2':  # relation contrasive loss
            align_loss = 0
            num_pairs = 0
            loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
            union_mask1 = mask1.sum(-1)  # S
            union_mask2 = mask2.sum(-1)  # S
            if match_style == 'within_episode':
                for s in range(S):
                    if union_mask1[s] <= 0 or union_mask2[s] <= 0:
                        continue
                    mask_tmp1 = mask1[s]  # N
                    mask_tmp2 = mask2[s]  # N
                    repres_tmp1 = repres1[s]  # N x C
                    repres_tmp2 = repres2[s]  # N x C
                    flag, align_loss_tmp = self.compute_contrastive_loss(repres_tmp1, mask_tmp1, repres_tmp2, mask_tmp2,
                                                                 sim_type, tau, loss_func, use_sampled_neg)
                    if flag == False:
                        continue
                    align_loss += align_loss_tmp
                    num_pairs += 1  # a pair of episodes
                    # if np.isnan(align_loss_tmp.cpu().detach().numpy()):
                    #     print('error')
            else:  # 'between_episode'
                for s1 in range(S):
                    for s2 in range(s1 + 1, S):
                        if union_mask1[s1] <= 0 or union_mask2[s2] <= 0:
                            continue
                        mask_tmp1 = mask1[s1]  # N
                        mask_tmp2 = mask2[s2]  # N
                        repres_tmp1 = repres1[s1]  # N x C
                        repres_tmp2 = repres2[s2]  # N x C
                        flag, align_loss_tmp = self.compute_contrastive_loss(repres_tmp1, mask_tmp1, repres_tmp2, mask_tmp2,
                                                                 sim_type, tau, loss_func, use_sampled_neg)
                        if flag == False:
                            continue
                        align_loss += align_loss_tmp
                        num_pairs += 1  # a pair of episodes
            align_loss = (align_loss / num_pairs) if num_pairs > 0 else torch.tensor(0.).cuda()
        else:
            raise NotImplementedError

        return align_loss

    def set_complement_itpl_kps_texts_info(self):
        self.itpl_kp_texts_info = {'texts': [], 'mask': []}

    def get_complement_itpl_kps_texts_info(self):
        '''
        texts: S x (N_path*N_knots)
        mask : S x (N_path*N_knots) x 1 (We only consider each interpolated keypoint has 1 assigned text!)
        if complement:
            1) if the mask is all zero, it means we only perform padding. In this case, 'texts' is [].
            2) if the mask is not all zero, it means we indeed perform text interpolation.
        if no complement:
            1) mask = None
        '''
        if len(self.itpl_kp_texts_info['mask']) > 0:
            masks_list = self.itpl_kp_texts_info['mask']
            self.itpl_kp_texts_info['mask'] = torch.stack(masks_list, dim=0)  # S x (N_path*N_knots) x 1
        else:
            self.itpl_kp_texts_info['mask'] = None
        return self.itpl_kp_texts_info

    def itpl_text_assignment_and_proto_compute(self, itpl_text_features, itpl_text_features_for_picking,
                                               texts_pool_per_episode, texts_pool_per_episode_mask,
                                               query_aux_kp_mask, query_aux_kp_repres=None, output_picked_texts=False, **kwargs):
        '''
        Assign interpolated text to visually interpolated kp & compute the text feature prototype
        :param itpl_text_features (used to compute prototype, Gradient required): N_path x T3 x D
        :param itpl_text_features_for_picking: N_path x T3 x D
        :param texts_pool_per_episode: a list of (N_path*T3) texts, where T3=N_reapeat * K_texts_per_path
        :param texts_pool_per_episode_mask: N_path x T3
        :param query_aux_kp_mask: B2 x N_aux = B2 x (N_path*N_knots)
        :param query_aux_kp_repres: B2 x N_aux x D
        :param output_picked_texts: False or True
        :param kwargs:
        :return:
        '''
        N_path, T3, D = itpl_text_features.shape
        B2, N_aux = query_aux_kp_mask.shape
        N_knots = N_aux // N_path  # N_aux=N_path*N_knots; N_knots visual interpolated kps.
        K_texts_per_path = self.cfg.DATASET.ITPL_TEXT_CORPUS_SETTING.NUM_TEXTS_PER_PATH
        # K_texts_per_path = 1  # TODO: If uncommented and set 1, used to test upper bound for GT interpolated texts
        N_repeat = T3 // K_texts_per_path  # T3=N_reapeat * K_texts_per_path, which includes repeatedly generated texts
        assert N_knots == 1, 'Currently we only support N_knots = 1'

        assign_strategy = self.cfg.TRAIN.TEXT_PROMPT_SETTING.ITPL_TEXT_SETTING.ASSIGN_STRATEGY
        # assign 1 text from the pool to a knot
        if assign_strategy == 'top1':
            rand_inds = torch.randint(N_repeat, [N_path])  # sample n-th repetition
            rand_inds *= K_texts_per_path  # convert into index in per-path level: n*K_texts_per_path+0
            path_inds = torch.arange(N_path * N_knots)
            itpl_kps_text_proto = itpl_text_features[path_inds, rand_inds, :] # (N_path*1) x D
            itpl_kps_texts_mask = texts_pool_per_episode_mask[path_inds, rand_inds]
            itpl_kps_texts_mask = itpl_kps_texts_mask.reshape(N_path, 1)  # (N_path*1) x 1
            if output_picked_texts:
                inds_tmp = path_inds * T3 + rand_inds  # global index
                picked_itpl_texts = list(map(lambda ind: texts_pool_per_episode[ind], inds_tmp))   # (N_path*1)
            else:
                picked_itpl_texts = None
        elif assign_strategy == 'rand':
            rand_inds = torch.randint(T3, [N_path])  # sample N_path*1 random numbers
            path_inds = torch.arange(N_path * N_knots)
            itpl_kps_text_proto = itpl_text_features[path_inds, rand_inds, :]  # (N_path*1) x D
            itpl_kps_texts_mask = texts_pool_per_episode_mask[path_inds, rand_inds]
            itpl_kps_texts_mask = itpl_kps_texts_mask.reshape(N_path, 1)  # (N_path*1) x 1
            if output_picked_texts:
                inds_tmp = path_inds * T3 + rand_inds                    # (N_path*1)
                picked_itpl_texts = list(map(lambda ind: texts_pool_per_episode[ind], inds_tmp))   # (N_path*1)
            else:
                picked_itpl_texts = None
        elif assign_strategy == 'corr':  # correlation via softmax over kp-text similarities
            assert query_aux_kp_repres is not None

            # 1. Compute cosine similarity: B2 x N_path x (N_knots*T3), note N_knots=1
            kp_repres_norm = torch.nn.functional.normalize(query_aux_kp_repres, p=2, dim=-1, eps=1e-12)  # B2 x N_aux x C
            text_repres_norm = torch.nn.functional.normalize(itpl_text_features_for_picking, p=2, dim=-1, eps=1e-12) # N_path x T3 x C
            kp_repres_norm = kp_repres_norm.reshape(B2, N_path, N_knots, D)
            text_repres_norm = text_repres_norm.unsqueeze(0).expand(B2, N_path, T3, D)  # B x N_path x T3 x C
            sim_matrix = (kp_repres_norm * text_repres_norm).sum(-1)  # B2 x N_path x T3 (N_knots=1)
            sim_matrix = torch.nn.functional.relu(sim_matrix)  # rule out negative similarities

            sim_matrix = sim_matrix.reshape(B2, N_path, N_repeat, K_texts_per_path)  # B2 x N_path x N_repeat x K_texts_per_path

            # averaging per-path similarity across B2 images
            query_aux_kp_mask_sum = query_aux_kp_mask.long().sum(0)  # N_aux (N_aux=N_path*N_knots=N_path)
            query_union_mask = (query_aux_kp_mask_sum > 0).long()    # N_aux (N_aux=N_path*N_knots=N_path)
            query_aux_kp_mask_sum2 = query_aux_kp_mask_sum + (1-query_union_mask)  # add 1 to avoid divide 0
            query_aux_kp_mask_sum2 = query_aux_kp_mask_sum2.reshape(N_aux, 1, 1)
            sim_matrix_avg = (sim_matrix * query_aux_kp_mask.unsqueeze(-1).unsqueeze(-1)).sum(0) / query_aux_kp_mask_sum2  # N_path x N_repeat x K_texts_per_path

            # 2. Pick interpolated texts
            tau = self.cfg.LOSS.DOMAIN_ALIGNMENT.SIMILARITY.TAU
            sim_matrix_avg = sim_matrix_avg.reshape(N_path, T3)
            sim_tmp = sim_matrix_avg - (100 * (1-texts_pool_per_episode_mask))  # N_path x T3, masking
            prob = torch.softmax(sim_tmp / tau, dim=-1)    # N_path x T3
            scores, text_inds = torch.max(prob, dim=-1)    # N_path, N_path (Note N_aux=N_path in our case)

            # 3. Gather sampled interpolated text features
            path_inds = torch.arange(N_path, dtype=torch.long).cuda()
            itpl_kps_text_proto = itpl_text_features[path_inds, text_inds, :]  # N_path x D
            match_mask = query_union_mask.clone()  # N_path (N_aux=N_path*N_knots=N_path)
            itpl_kps_texts_mask = match_mask.reshape(N_path, 1).cuda()  # N_path x 1

            if output_picked_texts:
                # transform to global index
                text_inds = text_inds + (path_inds * T3)  # N_path
                picked_itpl_texts = [texts_pool_per_episode[ind] for ind in text_inds]  # (N_path)
            else:
                picked_itpl_texts = None
        elif assign_strategy == 'corr2rej':  # sampling with false text rejection
            assert query_aux_kp_repres is not None

            # 1. Compute cosine similarity: B2 x N_path x (N_knots*T3), note N_knots=1
            kp_repres_norm = torch.nn.functional.normalize(query_aux_kp_repres, p=2, dim=-1, eps=1e-12)  # B2 x N_aux x C
            text_repres_norm = torch.nn.functional.normalize(itpl_text_features_for_picking, p=2, dim=-1, eps=1e-12) # N_path x T3 x C
            kp_repres_norm = kp_repres_norm.reshape(B2, N_path, N_knots, D)
            text_repres_norm = text_repres_norm.unsqueeze(0).expand(B2, N_path, T3, D)  # B x N_path x T3 x C
            sim_matrix = (kp_repres_norm * text_repres_norm).sum(-1)  # B2 x N_path x T3 (N_knots=1)
            sim_matrix = torch.nn.functional.relu(sim_matrix)  # rule out negative similarities

            sim_matrix = sim_matrix.reshape(B2, N_path, N_repeat, K_texts_per_path)  # B2 x N_path x N_repeat x K_texts_per_path

            # averaging per-path similarity across B2 images
            query_aux_kp_mask_sum = query_aux_kp_mask.long().sum(0)  # N_aux (N_aux=N_path*N_knots=N_path)
            query_union_mask = (query_aux_kp_mask_sum > 0).long()    # N_aux (N_aux=N_path*N_knots=N_path)
            query_aux_kp_mask_sum2 = query_aux_kp_mask_sum + (1-query_union_mask)  # add 1 to avoid divide 0
            query_aux_kp_mask_sum2 = query_aux_kp_mask_sum2.reshape(N_aux, 1, 1)
            sim_matrix_avg = (sim_matrix * query_aux_kp_mask.unsqueeze(-1).unsqueeze(-1)).sum(0) / query_aux_kp_mask_sum2  # N_path x N_repeat x K_texts_per_path

            # 2. Search range, i.e., top-K' (K'<=K_texts_per_path), and, similarity threshold
            cfg_tmp = self.cfg.TRAIN.TEXT_PROMPT_SETTING.ITPL_TEXT_SETTING.CORR2REJ
            topk = cfg_tmp.TOPK
            sim_thresh = kwargs['sim_thresh']  # cfg_tmp.SIM_THRESH

            # 3. sampling with false text rejection
            mask_tmp = sim_matrix_avg >= sim_thresh  # N_path x N_repeat x K_texts_per_path
            text_inds = torch.zeros(N_path, dtype=torch.long)
            match_mask = query_union_mask.clone()  # N_path (N_aux=N_path*N_knots=N_path)
            for p in range(N_path):
                # success_flag, sampled_n, sampled_k = progressive_sampling(mask_tmp[b, p], topk)
                success_flag, sampled_n, sampled_k = even_sampling(mask_tmp[p], topk)
                if success_flag == False:
                    match_mask[p] = 0
                    continue
                text_inds[p] = sampled_n * K_texts_per_path + sampled_k  # 0~T3 (N_repeat*K_texts_per_path)

            # 4. Gather sampled interpolated text features
            path_inds = torch.arange(N_path, dtype=torch.long)
            itpl_kps_text_proto = itpl_text_features[path_inds, text_inds, :]  # N_path x D
            itpl_kps_texts_mask = match_mask.reshape(N_path, 1).cuda()         # N_path x 1

            if output_picked_texts:
                # transform to global index
                text_inds = text_inds + (path_inds * T3)  # N_path
                picked_itpl_texts = [texts_pool_per_episode[ind] for ind in text_inds]  # (N_path)
            else:
                picked_itpl_texts = None
        else:
            raise NotImplementedError

        return itpl_kps_text_proto, itpl_kps_texts_mask, picked_itpl_texts

    def forward(self, supports_, queries_, support_kps_=None, support_kp_mask_=None, obj_texts_=((),), obj_texts_mask_=None,
                kps_texts_=((),), kps_texts_mask_=None, itpl_texts_pool_=((),), itpl_texts_pool_mask_=None, **kwargs):
        '''
        input arguments
        supports_ (image): S x B1 x 3 x H x W (S episodes' support images) or None or [[0], [0], ...]
        queries_ (image):  S x B2 x 3 x H x W (S episodes' query images)
        support_kps_: S x B1 x N_v x 2, ranges -1~1 (continuous) or None or [[0], [0], ...]
        support_kp_mask_: S x B1 x N_v

        obj_texts_: a list with S lists where each is an episode's T1 object texts (i.e. S x T1)
        obj_texts_mask_: tensor S x T1, or None
        kps_texts_: a list with S lists where each is an episode's N_t*T2 kps texts (i.e. S x (N_t*T2))
        kps_texts_mask_: tensor S x N_t x T2, or None

        itpl_texts_pool_: S x (N_path*T3), T3 is number of interpolated texts per path
        itpl_texts_pool_mask_: S x N_path x T3

        output arguments
        heatmaps_predict_list: a list of heatmaps, each is S x B2 x N x H x W (N=N_v or N_t)

        '''
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

        if self.eval_cost:
            time_start = time.time()

        # TODO: 1) Image and text features extraction
        if self.trunk == 'CLIP':
            # TODO: test CLIP's matching ability between images and texts
            # text_collection = ["a diagram", "a dog", "left ear", "triangles", "the horse", "cat", "cow", "sheep"]
            # text = clip.tokenize(text_collection).to('cuda')
            # logits_per_image, logits_per_text = self.encoder(queries_[0].cuda(), text)
            # probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
            # print("Label probs:", probs)

            # 1) clip's text tokenize: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 77
            in_texts_tokens = clip.tokenize(in_texts).cuda() if len(in_texts) > 0 else []

            # 2) extract text features: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 77 x d (CLIP)
            if len(in_texts_tokens) > 0:
                out_texts_features = self.encoder.encode_text(in_texts_tokens)
            else:
                out_texts_features = None

            # extract image features (after_proj and before_proj features)
            # Support+query: (S * (B1+B2)) x (1+H*W) x C or Query image: (S * B2) x (1+H*W) x C
            out_ims_features = self.encoder.encode_image(in_ims)

            # 3) freeze image and text features
            # out_ims_features = list(map(lambda x: x.detach(), out_ims_features))
            # if out_texts_features is not None:
            #     out_texts_features = list(map(lambda x: x.detach(), out_texts_features))

        elif self.trunk == 'BLIP':
            # TODO: test BLIP's matching ability between images and texts
            # text_collection = ["a diagram", "a dog", "left ear", "triangles", "the horse", "cat", "cow", "sheep"]
            # # image-text contrastive learning's similarity
            # sim = self.encoder(queries_[0].cuda(), text_collection, match_head='itc')
            # probs = torch.nn.functional.softmax(sim / self.encoder.temp, dim=1).detach().cpu().numpy()  # 5 x 8
            # print('The image and text\'s matching probs: \n', probs)
            # # image-text matching via a binary classifier
            # itm = self.encoder(queries_[0].cuda(), text_collection[-5:], match_head='itm')
            # itm_score = torch.nn.functional.softmax(itm, dim=1)[:, 1]  # 5 x 2 --> 5 x 1
            # itm_score = itm_score.detach().cpu().numpy()
            # print('The image and text\'s matching score: \n', itm_score)

            # 1) blip's text tokenize: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 35
            (in_texts_tokens, in_texts_tokens_mask) = self.encoder.tokenize_batch_texts(in_texts, device='cuda') if len(in_texts) > 0 else ([], [])

            # 2) extract text features: {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x 35 x d (BLIP)
            if len(in_texts_tokens) > 0:
                out_texts_features = self.encoder.encode_batch_texts(in_texts_tokens, in_texts_tokens_mask)
            else:
                out_texts_features = None

            # extract image features (after_proj and before_proj features)
            # Support+query: (S * (B1+B2)) x (1+H*W) x C or Query image: (S * B2) x (1+H*W) x C
            out_ims_features = self.encoder.encode_image(in_ims)

            # 3) freeze image and text features
            # out_ims_features = list(map(lambda x: x.detach(), out_ims_features))
            # if out_texts_features is not None:
            #     out_texts_features = list(map(lambda x: x.detach(), out_texts_features))
        else:
            raise NotImplementedError

        # TODO: upscale image feature map
        if self.feature_sr is not None:
            out_ims_features_0 = self.feature_map_upscale(out_ims_features[0])  # (S*B) x (1+r*H*r*W) x C
            out_ims_features = [out_ims_features_0]

        if self.eval_cost:
            self.time_meter1.update((time.time() - time_start) / S, n=S)  # sec / episode
            time_start = time.time()

        # TODO: 2) Feature adaptation
        feat_width = int(np.sqrt(out_ims_features[0].shape[1] - 1))
        assert (feat_width ** 2 + 1) == out_ims_features[0].shape[1], 'Feature resolution is not right.'
        if self.adaptation_net_type == None:
            adapted_ims_features = out_ims_features[0]
            adapted_texts_features = out_texts_features[0] if out_texts_features is not None else None
        elif self.adaptation_net_type == 'RESIDUAL_REFINE':  # (use after_proj and before_proj features)
            adapted_ims_features = self.vision_anet(out_ims_features[0], out_ims_features[1], h=feat_width, w=feat_width)
            adapted_texts_features = self.text_anet(out_texts_features[0], out_texts_features[1]) if out_texts_features is not None else None
        elif self.adaptation_net_type == 'RESIDUAL_REFINE2':  # (use after_proj features)
            adapted_ims_features = self.vision_anet(out_ims_features[0], out_ims_features[0], h=feat_width, w=feat_width)
            adapted_texts_features = self.text_anet(out_texts_features[0], out_texts_features[0]) if out_texts_features is not None else None
        else:
            raise NotImplementedError

        if self.eval_cost:
            self.time_meter2.update((time.time() - time_start) / S, n=S)  # sec / episode
            time_start = time.time()

        enable_alignment = (self.alignment_type is not None) and (kwargs.get('num_main_kps') is not None) and (kwargs.get('query_kps_') is not None)
        if enable_alignment == True:
            support_repres_list = []
            query_repres_list = []
            kps_text_proto_list = []
            kps_text_proto_mask_list = []

        heatmaps_list = []
        self.set_complement_itpl_kps_texts_info()  # clear
        for epi_ind in range(S):
            # TODO: 3) Keypoint prompt set building
            prompt_set = {'obj': [], 'text': [], 'image': []}
            # Parse encoded text features. We may have text features or not, depending on given prompts.
            # adapted_texts_features has size of {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x L x d
            if len(in_obj_texts) > 0:
                start_ind = epi_ind*T1
                end_ind   = (epi_ind+1)*T1
                if self.trunk == 'CLIP':  # for clip, text CLS token is at last
                    obj_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # T1 x D
                elif self.trunk == 'BLIP':  # for blip, text CLS token is at first
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
                if self.trunk == 'CLIP':  # for clip, text CLS token is at last
                    kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # (N_t*T2) x D
                elif self.trunk == 'BLIP':  # for blip, text CLS token is at first
                    kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind)][:, 0, :]  # (N_t*T2) x D
                # kps_text_proto = kps_text_CLS.reshape(N, T2, -1).mean(dim=1)  # N_t * D
                kps_text_mask_per_episode = kps_texts_mask_[epi_ind]  # N_t x T2
                kps_text_mask_per_episode_sum = kps_text_mask_per_episode.sum(dim=-1)  # N_t
                kps_text_proto = (kps_text_CLS.reshape(N_t, T2, -1) * kps_text_mask_per_episode.unsqueeze(-1)).sum(dim=1)  # N_t * D
                kps_text_mask_per_episode_tmp = (kps_text_mask_per_episode_sum <= 0).long() + kps_text_mask_per_episode_sum  # +1 to avoid dividing zero
                kps_text_proto /= kps_text_mask_per_episode_tmp.unsqueeze(-1)  # N_t * D
                kps_text_proto_mask = (kps_text_mask_per_episode_sum > 0).long()  # N_t
                prompt_set['text'] = kps_text_proto  # N_t * D
            else:
                kps_text_proto = []

            # Parse encoded image features. We definitely have query features, but may have support image features or not
            # Support+query: (S * (B1+B2)) x (1+h*w) x C or Query image: (S * B2) x (1+h*w) x C
            if B1 > 0:
                # support image (visual prompt): B1 x (1+h*w) x C
                support_im_tokens = adapted_ims_features[epi_ind*B_total: epi_ind*B_total+B1]
                support_kps = support_kps_[epi_ind]          # B1 x N_v x 2, ranges -1~1 (continuous)
                support_kp_mask = support_kp_mask_[epi_ind]  # B1 x N_v

                # query image: B2 x (1+h*w) x C
                query_im_tokens   = adapted_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]

                support_features = (support_im_tokens[:, 1:]).permute(0, 2, 1).reshape(B1, -1, feat_width, feat_width)  # B1 x C x h x w, remove CLS
                # TODO: Note we may bring in human bias when extracting keypoint representations here
                support_repres = self.visual_prompt_extraction(self.visual_prompt_extraction_type, support_features, support_kps, support_kp_mask, W)  # B1 x C x N_v
                avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N_v
                kps_visual_proto = avg_support_repres.transpose(1, 0)  # N_v x C
                prompt_set['image'] = kps_visual_proto  # TODO: Note not all visual proto are valid. Need masking when fusing

                if enable_alignment == True:  # hook features for domain alignment
                    support_repres_list.append(support_repres)  # B1 x C x N_v
            else:
                support_im_tokens = []
                kps_visual_proto = []
                # query image: B2 x (1+h*w) x C
                query_im_tokens = adapted_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]
            query_features = (query_im_tokens[:, 1:]).permute(0, 2, 1).reshape(B2, -1, feat_width, feat_width)  # B2 x C x h x w, remove CLS


            if kwargs.get('query_kps_') is not None:
                query_kps = kwargs['query_kps_'][epi_ind]          # B2 x N_gt x 2, ranges -1~1 (continuous)
                query_kp_mask = kwargs['query_kp_mask_'][epi_ind]  # B2 x N_gt
                query_repres = self.visual_prompt_extraction(self.visual_prompt_extraction_type, query_features, query_kps, query_kp_mask, W)  # B2 x C x N

                if (enable_alignment == True) and (self.use_query_kps == True):  # hook features for domain alignment
                    query_repres_list.append(query_repres)  # B2 x C x N_gt

            #======================================================================================================
            # TODO: Assign pseudo text to interpolated keypoint
            if (len(in_kps_texts) > 0) and (kwargs.get('query_kps_') is not None):
                N_gt = kwargs['query_kps_'].shape[-2]  # S x B2 x N_gt x 2
                assert N_t <= N_gt and N_v <= N_gt, 'the number of prompted kps should <= the number of annotated kps in query image'
                if N_t < N_gt:  # we need to complement itpl_kps_texts and itpl_kps_texts_mask
                    if len(in_itpl_texts) > 0:  # complement using itpl_texts_pool_
                        #------------------------------------------------------------------------------
                        # 1. Scheduler to use original CLIP features or adapted features for matching. This scheduler is to handle boostrap problem.
                        global_episode_cnt = kwargs['global_episode_cnt'] + epi_ind
                        itpl_cfg = self.cfg.TRAIN.TEXT_PROMPT_SETTING.ITPL_TEXT_SETTING
                        num_episode_using_origin_feat = itpl_cfg.SCHEDULER.NUM_EPISODE_ORIGIN_FEAT
                        # Using original CLIP features for bootstrapping (>0, e.g., 10000) or always use (-1)
                        if (global_episode_cnt < num_episode_using_origin_feat) or (num_episode_using_origin_feat==-1):
                            which_texts_feat_for_picking = 'original'
                            chosen_ims_features = out_ims_features[0]  # (S * (B1+B2)) x (1+h*w) x C or (S * B2) x (1+h*w) x C
                            # query image: B2 x (1+h*w) x C
                            query_im_tokens_tmp = chosen_ims_features[epi_ind*B_total+B1: (epi_ind + 1) * B_total]
                            query_features_tmp = (query_im_tokens_tmp[:, 1:]).permute(0, 2, 1).reshape(B2, -1, feat_width, feat_width)  # B2 x C x h x w, remove CLS
                            # N_aux = N_gt - N_t
                            query_aux_kp_tmp = query_kps[:, N_t:]       # B2 x N_aux x 2, ranges -1~1 (continuous)
                            query_aux_kp_mask = query_kp_mask[:, N_t:]  # B2 x N_aux
                            query_aux_kp_repres = self.visual_prompt_extraction(self.visual_prompt_extraction_type, query_features_tmp, query_aux_kp_tmp, query_aux_kp_mask, W)  # B2 x C x N_aux
                            query_aux_kp_repres = query_aux_kp_repres.permute(0, 2, 1)  # B2 x N_aux x C
                        else:  # Using adapted features. Note we already extracted query_repres in above code (L1183).
                            which_texts_feat_for_picking = 'adapted'
                            # N_aux = N_gt - N_t
                            query_aux_kp_mask = query_kp_mask[:, N_t:]  # B2 x N_aux
                            query_aux_kp_repres = query_repres[:, :, N_t:]  # B2 x C x N_aux
                            query_aux_kp_repres = query_aux_kp_repres.permute(0, 2, 1)  # B2 x N_aux x C

                        # 2. Adaptive threshold for similarity
                        ada_t0 = itpl_cfg.CORR2REJ.SIM_THRESH
                        if itpl_cfg.SCHEDULER.ADAPTIVE_SIM_THRESH == False:
                            sim_thresh = ada_t0
                        else:
                            if global_episode_cnt < num_episode_using_origin_feat:
                                sim_thresh = ada_t0
                            else:
                                ada_up = itpl_cfg.SCHEDULER.SIM_THRESH_UP  # upper bound
                                ada_x = itpl_cfg.SCHEDULER.NUM_EPISODE_UP
                                ada_t = (global_episode_cnt - num_episode_using_origin_feat) / ada_x * (ada_up-ada_t0) + ada_t0
                                sim_thresh = min(ada_t, ada_up)
                        # ------------------------------------------------------------------------------

                        # adapted_texts_features has size of {S*T1 + S*(N_t*T2) + S*(N_path*T3)} x L x d
                        base_ind = len(in_obj_texts) + len(in_kps_texts)
                        start_ind = base_ind + epi_ind*N_path*T3
                        end_ind = base_ind + (epi_ind+1)*N_path*T3
                        if self.trunk == 'CLIP':  # for clip, text CLS token is at last
                            kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # (N_path*T3) x D
                            if which_texts_feat_for_picking == 'adapted':
                                texts_feat_for_picking = kps_text_CLS
                            else:  # original
                                texts_feat_for_picking = out_texts_features[0][torch.arange(start_ind, end_ind), in_texts_tokens[start_ind:end_ind].argmax(dim=-1)]  # (N_path*T3) x D
                        elif self.trunk == 'BLIP':  # for blip, text CLS token is at first
                            kps_text_CLS = adapted_texts_features[torch.arange(start_ind, end_ind)][:, 0, :]  # (N_path*T3) x D
                            if which_texts_feat_for_picking == 'adapted':
                                texts_feat_for_picking = kps_text_CLS
                            else:  # original
                                texts_feat_for_picking = out_texts_features[0][torch.arange(start_ind, end_ind)][:, 0, :]  # (N_path*T3) x D
                        else:
                            raise NotImplementedError

                        # TODO: Algorithm to assign interpolated texts: (N_path*T3) x D --> (N_path*N_knots) x D
                        # TODO: And compute interpolated text features' prototype
                        kps_text_CLS = kps_text_CLS.reshape(N_path, T3, -1)  # N_path x T3 x D
                        texts_feat_for_picking = texts_feat_for_picking.reshape(N_path, T3, -1)  # N_path x T3 x D
                        texts_pool_per_episode = itpl_texts_pool_[epi_ind]   # (N_path*T3)
                        texts_pool_per_episode_mask = itpl_texts_pool_mask_[epi_ind]  # N_path x T3

                        itpl_results = self.itpl_text_assignment_and_proto_compute(
                            kps_text_CLS, texts_feat_for_picking, texts_pool_per_episode, texts_pool_per_episode_mask,
                            query_aux_kp_mask,
                            query_aux_kp_repres,
                            output_picked_texts=False,
                            sim_thresh=sim_thresh,
                        )
                        itpl_kps_text_proto = itpl_results[0]
                        itpl_kps_texts_mask = itpl_results[1]
                        itpl_kps_texts      = itpl_results[2]
                        self.itpl_kp_texts_info['texts'].append(itpl_kps_texts)  # record assigned texts
                    else:  # complement via 0
                        # Note N_aux = N_gt-N_t = N_path*N_knots
                        itpl_kps_text_proto = torch.zeros(N_gt-N_t, kps_text_proto.shape[-1]).cuda()  # N_aux x D
                        itpl_kps_texts_mask = torch.zeros(N_gt-N_t, 1).cuda()  # N_aux x 1
                    
                    # Update text prototypes by complementing: (N_t+N_path*N_knots) x D = N_gt x D
                    kps_text_proto = torch.cat([kps_text_proto, itpl_kps_text_proto], dim=0)
                    kps_text_proto_mask = torch.cat([kps_text_proto_mask, itpl_kps_texts_mask.squeeze()], dim=0)
                    prompt_set['text'] = kps_text_proto

                    self.itpl_kp_texts_info['mask'].append(itpl_kps_texts_mask)  # record assigned texts mask
            # ======================================================================================================

            if (enable_alignment == True) and (len(in_kps_texts) > 0):  # hook features for domain alignment
                kps_text_proto_list.append(kps_text_proto)            # N_t * D or (N_t + N_aux) * D
                kps_text_proto_mask_list.append(kps_text_proto_mask)  # N_t or (N_t + N_aux)

            # TODO: 4) Feature correlation + heatmap generation
            # heatmaps induced by corresponding prompts are as follows:
            heatmaps = self.correlation_decoding(prompt_set, query_features)

            # TODO: 5) Heatmap upscaling
            # `` heatmaps_set = {'obj': [], 'text': [], 'image': []} "
            heatmaps_set = self.heatmap_upscale(heatmaps, prompt_set)
            heatmaps_list.append(heatmaps_set)

        # TODO: Loss for domain alignment
        if enable_alignment == True:
            N_main = kwargs['num_main_kps']  # N_main
            N_gt = kwargs['query_kps_'].shape[-2]  # S x B2 x N_gt x 2
            N_aux  = N_gt - N_main

            # prepare visual main keypoint features (average)
            vm_repres_list = []
            vm_repres_mask_list = []
            va_repres_list = []
            va_repres_mask_list = []
            if len(support_repres_list) > 0:
                align_s_repres = torch.stack(support_repres_list, dim=0)     # S x B1 x C x N_v
                vm_repres_list.append(align_s_repres[:, :, :, :N_main])      # S x B1 x C x N_main
                vm_repres_mask_list.append(support_kp_mask_[:, :, :N_main])  # S x B1 x N_main

                if self.v_t_align_itpl:  # prepare visual interpolated kps features
                    assert N_aux > 0, 'Aligning interpolated kps and texts requires N_itpl > 0'
                    # in each episode, choose one support image with max number of valid auxiliary kps.
                    itpl_s_repres = align_s_repres[:, :, :, N_main:]  # S x B1 x C x N_aux
                    itpl_kp_mask = support_kp_mask_[:, :, N_main:]    # S x B1 x N_aux
                    va_repres_list.append(itpl_s_repres)
                    va_repres_mask_list.append(itpl_kp_mask)

            if self.use_query_kps == True:
                align_q_repres = torch.stack(query_repres_list, dim=0)  # S x B2 x C x N_gt
                align_q_masks  = kwargs['query_kp_mask_']               # S x B2 x N_gt
                vm_repres_list.append(align_q_repres[:, :, :, :N_main]) # S x B2 x C x N_main
                vm_repres_mask_list.append(align_q_masks[:, :, :N_main])# S x B2 x N_main
                va_repres_list.append(align_q_repres[:, :, :, N_main:]) # S x B2 x C x N_aux
                va_repres_mask_list.append(align_q_masks[:, :, N_main:])# S x B2 x N_aux

            if len(vm_repres_list) > 0:
                vm_repres = torch.cat(vm_repres_list, dim=1)            # S x (B1+B2) x C x N_main
                vm_repres_mask = torch.cat(vm_repres_mask_list, dim=1)  # S x (B1+B2) x N_main
                # S x N_main x C, S x N_main
                vm_repres, vm_repres_mask = self.get_instance_or_proto_features_for_CL(vm_repres, vm_repres_mask, use_proto=self.use_proto_main)
            else:
                vm_repres, vm_repres_mask = None, None
            if len(va_repres_list) > 0:
                va_repres = torch.cat(va_repres_list, dim=1)            # S x (B1+B2) x C x N_aux
                va_repres_mask = torch.cat(va_repres_mask_list, dim=1)  # S x (B1+B2) x N_aux
                # S x N_aux x C, S x N_aux
                va_repres, va_repres_mask = self.get_instance_or_proto_features_for_CL(va_repres, va_repres_mask, use_proto=self.use_proto_itpl)
            else:
                va_repres, va_repres_mask = None, None

            # prepare textual main keypoint features
            if len(kps_text_proto_list) > 0:
                align_t_repres = torch.stack(kps_text_proto_list, dim=0)  # S x N x C
                align_t_repres_mask = torch.stack(kps_text_proto_mask_list, dim=0)  # S x N
                tm_repres = align_t_repres[:, :N_main, :]         # S x N_main x C
                tm_repres_mask = align_t_repres_mask[:, :N_main]  # S x N_main

                if self.v_t_align_itpl:  # prepare textual interpolated features
                    ta_repres = align_t_repres[:, N_main:, :]         # S x N_aux x C
                    ta_repres_mask = align_t_repres_mask[:, N_main:]  # S x N_aux

            # 1. visual-textual alignment loss
            if (self.v_t_align == True) and (len(vm_repres_list) > 0) and (len(kps_text_proto_list) > 0):
                if self.sg_textual == True:  # Stop gradient to align visual features towards textual features
                    tm_repres_t = tm_repres.detach()
                else:
                    tm_repres_t = tm_repres
                v_t_align_loss = self.domain_alignment(vm_repres, vm_repres_mask, tm_repres_t, tm_repres_mask,
                                        self.alignment_type, self.sim_type, self.tau, match_style='within_episode')
            else:
                v_t_align_loss = None
            # 2. visual-visual alignment loss
            if (self.v_v_align == True) and (len(vm_repres_list) > 0):
                v_v_align_loss = self.domain_alignment(vm_repres, vm_repres_mask, vm_repres, vm_repres_mask,
                                        self.alignment_type, self.sim_type, self.tau, match_style='between_episode')
            else:
                v_v_align_loss = None
            # 3. textual-textual alignment loss
            if (self.t_t_align == True) and (len(kps_text_proto_list) > 0):
                t_t_align_loss = self.domain_alignment(tm_repres, tm_repres_mask, tm_repres, tm_repres_mask,
                                        self.alignment_type, self.sim_type, self.tau, match_style='between_episode')
            else:
                t_t_align_loss = None
            # 4. visual-textual alignment loss for interpolated kps and texts
            if (self.v_t_align_itpl == True) and (len(va_repres_list) > 0) and (len(kps_text_proto_list) > 0):
                if self.sg_textual == True:  # Stop gradient to align visual features towards textual features
                    ta_repres_t = ta_repres.detach()
                else:
                    ta_repres_t = ta_repres
                v_t_itpl_loss = self.domain_alignment(va_repres, va_repres_mask, ta_repres_t, ta_repres_mask,
                                        self.alignment_type, self.sim_type, self.tau, match_style='within_episode')
            else:
                v_t_itpl_loss = None
        else:
            v_t_align_loss, v_v_align_loss, t_t_align_loss = None, None, None
            v_t_itpl_loss = None

        if self.eval_cost:
            self.time_meter3.update((time.time() - time_start) / S, n=S)  # sec / episode

        return (heatmaps_list, v_t_align_loss, v_v_align_loss, t_t_align_loss, v_t_itpl_loss)

    def get_instance_or_proto_features_for_CL(self, repres, repres_mask, use_proto=True):
        '''
        :param repres: S x B x C x N
        :param repres_mask: S x B x N
        :param use_proto: True or False
        :return: pick_feat: S x N x C
                 pick_feat_mask:S x N
        '''
        S, B, _ = repres_mask.shape
        if use_proto == True:  # compute ``mean features for CL"
            repres_mask = repres_mask.sum(1)     # S x N
            mask_tmp = (repres_mask > 0).long()  # S x N, set 1 or 0 for keypoint mask
            repres_mask = repres_mask * mask_tmp + (1 - mask_tmp) * B  # avoid divide 0, S x N
            repres = repres.sum(1)  # S x C x N
            repres /= repres_mask.unsqueeze(1)
            pick_feat = repres.permute(0, 2, 1)  # S x N x C
            pick_feat_mask = mask_tmp            # S x N
        else:  # compute ``instance features for CL"
            ind_max_valid = repres_mask.sum(-1).max(-1)[1]   # S
            ind_tmp = torch.arange(0, S)  # S
            pick_feat = repres[ind_tmp, ind_max_valid]  # S x C x N
            pick_feat = pick_feat.permute(0, 2, 1)      # S x N x C
            pick_feat_mask = repres_mask[ind_tmp, ind_max_valid]  # S x N
        return pick_feat, pick_feat_mask

def get_openkd_model(cfg, **kwargs):
    openkd_model = OpenKDModel(cfg)
    openkd_model.init_weights(pretrained=cfg.MODEL.PRETRAINED)

    return openkd_model