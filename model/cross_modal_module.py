import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.paragraph_module import ParagraphModule
from model.image_module import ImageModule
import math

class CrossModalModule(BaseModule):
    def __init__(self, config, encoder):
        super().__init__(config)
        if not config.remove_para:
            self.paragraph_module = ParagraphModule(config, encoder)
        if not config.remove_img:
            self.image_module = ImageModule(config)
        if config.para_pooling == 'attention':
            self.para_Q = nn.ParameterList([nn.Parameter(self.weight_init(config.attention_dim, 'uniform')) for i in range(config.head_num)])
            self.para_K = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.para_V = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.module_embedding_dim // config.head_num) for i in range(config.head_num)])
        if config.img_pooling == 'attention':
            self.img_Q = nn.ParameterList([nn.Parameter(self.weight_init(config.attention_dim, 'uniform')) for i in range(config.head_num)])
            self.img_K = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.img_V = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.module_embedding_dim // config.head_num) for i in range(config.head_num)])
        if not config.without_cross_modal_attention:
            self.cross_Q1 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.cross_Q2 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.cross_K1 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.cross_K2 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.cross_V1 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.module_embedding_dim // config.head_num) for i in range(config.head_num)])
            self.cross_V2 = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.module_embedding_dim // config.head_num) for i in range(config.head_num)])
            self.norm1 = nn.LayerNorm(config.module_embedding_dim)
            self.norm2 = nn.LayerNorm(config.module_embedding_dim)
            self.dropout1 = nn.Dropout(config.dropout_rate)
            self.dropout2 = nn.Dropout(config.dropout_rate)
    
    def cross_modal_module(self, para_embeds, para_masks, img_embeds, img_masks, Q1, K1, V1, Q2, K2, V2, norm1, norm2, dropout1, dropout2):
        para_img_embeds = self.cross_modal_attention(para_embeds, para_masks, img_embeds, img_masks, Q1, K1, V1)
        img_para_embeds = self.cross_modal_attention(img_embeds, img_masks, para_embeds, para_masks, Q2, K2, V2)
        para_embeds = norm1(para_embeds+dropout1(para_img_embeds))
        img_embeds = norm2(img_embeds+dropout2(img_para_embeds))
        return para_embeds, img_embeds
    
    def forward(self, batch):
        para_embeds, img_embeds = None, None
        if not self.config.remove_para:
            para_embeds = self.paragraph_module(batch)
            para_masks = batch['para_masks']
        if not self.config.remove_img:
            img_embeds = self.image_module(batch)
            img_masks = batch['img_masks']
        if not self.config.remove_para and not self.config.remove_img:
            if not self.config.without_cross_modal_attention:
                para_embeds, img_embeds = self.cross_modal_module(para_embeds, para_masks, img_embeds, img_masks, self.cross_Q1, self.cross_K1, self.cross_V1, self.cross_Q2, self.cross_K2, self.cross_V2, self.norm1, self.norm2, self.dropout1, self.dropout2)
        # para_embeds & img_embeds: [batch_size, seq_len, module_embedding_dim]
        if not self.config.remove_para:
            if self.config.para_pooling == 'attention':
                para_embeds = self.multihead_attention_pooling(self.para_Q, para_embeds, para_masks, self.para_K, self.para_V)
            if self.config.para_pooling == 'average':
                para_embeds = self.average_pooling(para_embeds, para_masks)
            if self.config.para_pooling == 'maximum':
                para_embeds = self.max_pooling(para_embeds, para_masks)
        if not self.config.remove_img:
            if self.config.img_pooling == 'attention':
                img_embeds = self.multihead_attention_pooling(self.img_Q, img_embeds, img_masks, self.img_K, self.img_V)
            if self.config.img_pooling == 'average':
                img_embeds = self.average_pooling(img_embeds, img_masks)
            if self.config.img_pooling == 'maximum':
                para_embeds = self.max_pooling(img_embeds, img_masks)
        # para_embeds & img_embeds: [batch_size, module_embedding_dim]
        return para_embeds, img_embeds