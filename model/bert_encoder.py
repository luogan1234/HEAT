import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from transformers import BertModel

class BERTEncoder(BaseModule):        
    def __init__(self, config):
        super().__init__(config)
        if config.language == 'zh':
            bert = BertModel.from_pretrained('bert-base-chinese')
        if config.language == 'en':
            bert = BertModel.from_pretrained('bert-base-uncased')
        bert.to(config.device)
        if config.text_encoder == 'bert':
            bert.train()
            self.bert = bert
        else:
            bert.eval()
            for p in bert.parameters():
                p.requires_grad = False
            BERTEncoder.bert = bert
    
    def encode_layer(self, x):
        if self.config.text_encoder == 'bert':
            outputs = self.bert(x, attention_mask=(x>0))
            h = outputs.last_hidden_state
        else:
            with torch.no_grad():
                outputs = BERTEncoder.bert(x, attention_mask=(x>0))
                h = outputs.last_hidden_state
                h = h.detach()
        return h
    
    def forward(self, inputs):
        h = self.encode_layer(inputs)  # h: [batch_size, seq_len, para_embedding_dim]
        return h