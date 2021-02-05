import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.name_module import NameModule
from model.cross_modal_module import CrossModalModule
from model.bilstm_encoder import BiLSTMEncoder
from model.bert_encoder import BERTEncoder

class Model(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        if config.text_encoder == 'bilstm':
            encoder = BiLSTMEncoder(config)
        if config.text_encoder in ['bert', 'bert_freeze']:
            encoder = BERTEncoder(config)
        if not config.remove_name:
            self.name_module = NameModule(config, encoder)
        if not config.remove_para or not config.remove_img:
            self.cross_modal_module = CrossModalModule(config, encoder)
        if config.fusion == 'concatenation':
            self.fc = nn.Linear(config.feature_dim, config.label_num)
        elif config.fusion == 'attention':  # single head (only 3 modules)
            self.ent_Q = nn.Parameter(self.weight_init(config.attention_dim, 'uniform'))
            self.ent_K = nn.Linear(config.module_embedding_dim, config.attention_dim)
            self.ent_V = nn.Linear(config.module_embedding_dim, config.module_embedding_dim)
            self.fc = nn.Linear(config.module_embedding_dim, config.label_num)
        else:
            self.fc = nn.ModuleList([nn.Linear(config.module_embedding_dim, config.label_num) for i in range(3)])
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def final_layer(self, embeds_list):
        if self.config.fusion == 'concatenation':
            embeds = [embed for embed in embeds_list if embed is not None]
            embeds = torch.cat(embeds, 1)
            self.embeddings = embeds
            outs = self.fc(self.dropout(embeds))
        elif self.config.fusion == 'attention':
            # single head, because only 3 modules
            embeds = [embed.unsqueeze(1) for embed in embeds_list if embed is not None]
            embeds = torch.cat(embeds, 1)  # [batch_size, module_num, module_embedding_dim]
            mask = torch.ones(embeds.size(0), embeds.size(1), dtype=torch.bool)
            embeds = self.attention_pooling(self.ent_Q, embeds, mask, self.ent_K, self.ent_V)
            outs = self.fc(self.dropout(embeds))
        else:
            embeds = []
            for i, embed in enumerate(embeds_list):
                if embed is not None:
                    embeds.append(self.fc[i](self.dropout(embed)).unsqueeze(1))
            embeds = torch.cat(embeds, 1)
            if self.config.fusion == 'average':
                outs = torch.mean(embeds, 1)
            if self.config.fusion == 'maximum':
                outs = torch.max(embeds, 1)[0]
        return outs
    
    def forward(self, batch):
        name_embeds, para_embeds, img_embeds = None, None, None
        if not self.config.remove_name:
            name_embeds = self.name_module(batch)
        if not self.config.remove_para or not self.config.remove_img:
            para_embeds, img_embeds = self.cross_modal_module(batch)
        outputs = self.final_layer([name_embeds, para_embeds, img_embeds])
        return outputs