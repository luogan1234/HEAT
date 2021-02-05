import torch
import torch.nn as nn
import torch.nn.functional as F
from model.base_module import BaseModule

class NameModule(BaseModule):
    def __init__(self, config, encoder):
        super().__init__(config)
        self.encoder = encoder
    
    def forward(self, batch):
        x = batch['name_inputs']
        h = self.encoder(x)
        if self.config.name_pooling == 'average':
            name_embeds = self.average_pooling(h, x>0)
        if self.config.name_pooling == 'maximum':
            name_embeds = self.max_pooling(h, x>0)
        return name_embeds  # [batch_size, module_embedding_dim]