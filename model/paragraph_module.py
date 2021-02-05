import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
import math

class ParagraphModule(BaseModule):
    def __init__(self, config, encoder):
        super().__init__(config)
        self.encoder = encoder
        if not self.config.without_token_attention:
            self.token_Q = nn.ParameterList([nn.Parameter(self.weight_init(config.attention_dim, 'uniform')) for i in range(config.head_num)])
            self.token_K = nn.ModuleList([nn.Linear(config.para_embedding_dim, config.attention_dim) for i in range(config.head_num)])
            self.token_V = nn.ModuleList([nn.Linear(config.para_embedding_dim, config.module_embedding_dim // config.head_num) for i in range(config.head_num)])
    
    def forward(self, batch):
        x = batch['para_inputs']
        input_size = x.size()
        batch_size, para_num = input_size[:2]
        x = x.view(batch_size*para_num, -1)  # [batch_size*para_num, seq_len]
        h = self.encoder(x)
        if not self.config.without_token_attention:
            para_embeds = self.multihead_attention_pooling(self.token_Q, h, x>0, self.token_K, self.token_V)
        else:
            if self.config.text_encoder in ['bert', 'bert_freeze']:
                para_embeds = h[:, 0, :]
            if self.config.text_encoder in ['bilstm']:
                para_embeds = h[:, -1, :]
        para_embeds = para_embeds.view(batch_size, para_num, -1)  # [batch_size, para_num, module_embedding_dim]
        return para_embeds