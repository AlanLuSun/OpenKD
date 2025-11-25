from .med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from .blip import create_vit, init_tokenizer, load_checkpoint

class BLIP_For_KD(nn.Module):
    def __init__(self,
                 med_config = 'network/blip_kd/med_config.json',
                 image_size = 224,  # blip pre-trained with size 224
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 embed_dim = 256,
                 vision_layer_to_tune: int = -1,  # -1, disable; 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
                 text_layer_to_tune: int = -1  # -1, disable; 0, no layer to tune (all freeze); 1, proj to tune; >=2, the last n layers to tune
                 ):
        """
        blip's default patch size is 16 x 16, input image reso is 224, patch grids 14 x 14=196
        visual_dim = textual_dim = 768, projected dim is 256, namely 768-->256

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        text_width = self.text_encoder.config.hidden_size  # 768
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))  # finally T=0.0121 learned by BLIP

        self.vision_layer_to_tune = vision_layer_to_tune
        self.text_layer_to_tune = text_layer_to_tune
        self.image_size = image_size

    def encode_image(self, image):
        # return image features
        x = self.visual_encoder(image)  # B x L x D (before-projection tokens, where L=1+h*w)
        if self.vision_layer_to_tune == 1:
            x = x.detach()

        x2 = self.vision_proj(x)  # B x L x D (after-projection tokens, where L=1+h*w)
        if self.vision_layer_to_tune == 0:  # no layer to tune (all freeze)
            x2 = x2.detach()

        return x2, x

    def tokenize_batch_texts(self, texts, device='cuda'):
        texts_output = self.tokenizer(texts, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(device)

        # text_tokens: N x 35; text_tokens_mask: N x 35
        return texts_output.input_ids, texts_output.attention_mask

    def encode_batch_texts(self, text_tokens, text_tokens_mask):
        # for blip, text CLS token is at first

        text_features_out = self.text_encoder(text_tokens, attention_mask=text_tokens_mask, return_dict=True, mode='text')
        x = text_features_out.last_hidden_state  # N x L x D (before-projection tokens)
        if self.text_layer_to_tune == 1:
            x = x.detach()

        x2 = self.text_proj(x)  # N x L x D (after-projection tokens)
        if self.text_layer_to_tune == 0:  # no layer to tune (all freeze)
            x2 = x2.detach()

        return x2, x

    def encode_text(self, text, device='cuda'):  # it can only encode single text.
        # return text features
        x = self.tokenizer(text, return_tensors="pt").to(device)
        text_output = self.text_encoder(x.input_ids, attention_mask=x.attention_mask,
                                        return_dict=True, mode='text')
        return text_output.last_hidden_state

    def encode_multimodal(self, image, text, device='cuda'):
        # return multimodel features
        text = self.tokenizer(text, return_tensors="pt").to(device)
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text.input_ids[:, 0] = self.tokenizer.enc_token_id
        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )
        return output.last_hidden_state

    def forward(self, image, caption, match_head='itm'):
        '''
            compute image-text matching score or contrastive similarity
        '''
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image.device)


        if match_head=='itm':  # the number of images and texts should be same to form pairs for binary classification
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       return_dict = True,
                                      )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])
            return itm_output

        elif match_head=='itc':
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)  # for blip, text CLS token is at first

            sim = image_feat @ text_feat.t()
            return sim

def load(pretrained='',**kwargs):
    model = BLIP_For_KD(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model
