import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.att_d = 1.0 / math.sqrt(config.attention_dim)
    
    def weight_init(self, shape, mode):
        if isinstance(shape, int):
            shape = [shape]
        assert mode in ['uniform', 'normal'] and len(shape) > 0
        w = torch.empty(shape)
        std = 1 / math.sqrt(shape[0])
        if mode == 'uniform':
            if len(shape) == 1:
                nn.init.uniform_(w, -std, std)
            else:
                nn.init.xavier_uniform_(w)
        if mode == 'normal':
            if len(shape) == 1:
                nn.init.normal_(w, 0, std)
            else:
                nn.init.xavier_normal_(w)
        return w
    
    def multihead_attention_pooling(self, Q, KV, mask, Wk, Wv):
        '''
        Q: [head_num, attention_dim], KV: [batch_size, seq_len, embedding_dim]
        mask: [batch_size, seq_len], Wk: [head_num, embedding_dim, attention_dim]
        Wv: [head_num, embedding_dim, embedding_dim/head_num]
        '''
        Vs = []
        for i in range(self.config.head_num):
            alphas = torch.matmul(Wk[i](KV), Q[i])*self.att_d  # [batch_size, seq_len]
            mask[~torch.any(mask, 1)] = True  # avoid no valid KV
            alphas[~mask] = float('-inf')
            alphas = F.softmax(alphas, 1)
            V = torch.bmm(alphas.unsqueeze(1), Wv[i](KV)).squeeze(1)
            Vs.append(V)
        outs = torch.cat(Vs, 1)  # [batch_size, embedding_dim]
        return outs
    
    def attention_pooling(self, Q, KV, mask, Wk, Wv):
        '''
        Q: [attention_dim], KV: [batch_size, seq_len, embedding_dim]
        mask: [batch_size, seq_len], Wk: [embedding_dim, attention_dim]
        Wv: [embedding_dim, embedding_dim]
        '''
        alphas = torch.matmul(Wk(KV), Q)*self.att_d  # [batch_size, seq_len]
        mask[~torch.any(mask, 1)] = True  # avoid no valid KV
        alphas[~mask] = float('-inf')
        alphas = F.softmax(alphas, 1)
        outs = torch.bmm(alphas.unsqueeze(1), Wv(KV)).squeeze(1)  # [batch_size, embedding_dim]
        return outs
    
    def average_pooling(self, h, mask):
        '''
        h: [batch_size, seq_len, embedding_dim], mask: [batch_size, seq_len]
        '''
        outs = []
        for i in range(h.size(0)):
            h[i][mask[i]==0] = 0
            outs.append(h[i].sum(0)/mask[i].sum())
        outs = torch.stack(outs)  # [batch_size, embedding_dim]
        return outs
    
    def max_pooling(self, h, mask):
        '''
        h: [batch_size, seq_len, embedding_dim], mask: [batch_size, seq_len]
        '''
        outs = []
        for i in range(h.size(0)):
            h[i][mask[i]==0] = float('-inf')
            h_max = torch.max(h[i], 0)[0]
            outs.append(h_max)
        outs = torch.stack(outs)  # [batch_size, embedding_dim]
        return outs
    
    def cross_modal_attention(self, Q, mask_q, KV, mask_kv, Wq, Wk, Wv):
        '''
        Q & KV: [batch_size, seq_len, embedding_dim], mask_q & mask_kv: [batch_size, seq_len]
        Wq & Wk: [head_num, embedding_dim, attention_dim], Wv: [head_num, embedding_dim, embedding_dim/head_num]
        '''
        Vs = []
        for i in range(self.config.head_num):
            alphas = torch.bmm(Wk[i](KV), Wq[i](Q).transpose(1, 2))*self.att_d  # [batch_size, seq_len_kv, seq_len_q]
            mask_kv[~torch.any(mask_kv, 1)] = True
            alphas[~mask_kv] = float('-inf')
            alphas = F.softmax(alphas, 1).transpose(1, 2)  # [batch_size, seq_len_q, seq_len_kv]
            V = torch.bmm(alphas, Wv[i](KV))  # [batch_size, seq_len_q, embedding_dim/head_num]
            V[~mask_q] = 0
            Vs.append(V)
        outs = torch.cat(Vs, 2)  # [batch_size, seq_len_q, embedding_dim]
        return outs
    
    def forward(self, inputs):
        raise NotImplementedError