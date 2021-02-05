import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule

class BiLSTMEncoder(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.word_embedding = nn.Embedding(config.vocab_num, config.word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(config.word_embedding_dim, config.para_embedding_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def encode_layer(self, inputs, x):
        h, _ = self.lstm(x)
        return h
    
    def forward(self, inputs):
        x = self.word_embedding(inputs)
        h = self.encode_layer(inputs, x)  # [batch_size, seq_len, para_embedding_dim]
        h = self.dropout(h)
        return h