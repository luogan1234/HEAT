import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module import BaseModule
import math

class ImageModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.fc = nn.Linear(config.img_embedding_dim, config.module_embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, batch):
        x = batch['img_inputs']
        img_embeds = self.fc(self.dropout(x))  # [batch_size, seq_len, module_embedding_dim]
        return img_embeds