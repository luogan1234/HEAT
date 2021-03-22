import numpy as np
import torch
import math

class Config:
    def __init__(self, dataset, task, text_encoder, remove_info, module_info, consistency, labeled_num, seed, cpu):
        self.dataset = dataset
        self.task = task
        self.text_encoder = text_encoder
        self.remove_info = remove_info
        self.module_info = module_info
        self.remove_name, self.remove_para, self.remove_img = remove_info
        self.without_token_attention, self.without_cross_modal_attention = module_info
        self.consistency = consistency
        self.labeled_num = labeled_num
        self.seed = seed
        self.device = 'cpu' if cpu else 'cuda'
        
        if self.dataset in ['meituan_food']:
            self.language = 'zh'
            self.vocab_num = 21128  # bert-base-chinese
        if self.dataset in ['type_net', 'med_mentions', 'flower']:
            self.language = 'en'
            self.vocab_num = 30522  # bert-base-uncased
        if self.dataset in ['flower']:
            self.loss_fn = 'ce'
        else:
            self.loss_fn = 'bce'
        assert self.language, 'Need to provide the language information in config.py for new datasets.'
        
        self.name_len = 20  # include [CLS] & [SEP]
        self.para_num = 10
        self.para_len = 128  # include [CLS] & [SEP]
        self.img_num = 10
        self.dropout_rate = 0.1
        self.con_dropout_rate = 0.2
        self.con_temperature = 0.5
        self.con_threshold = 0.8
        self.con_weight = 1.0
        self.con_warmup = 1000
        self.attention_dim = 64
        self.max_steps = 2500
        
        self.name_pooling = 'average'
        self.para_pooling = 'attention'
        self.img_pooling = 'attention'
        self.head_num = 4
        self.fusion = 'concatenation'
        assert self.name_pooling in ['average', 'maximum']
        assert self.para_pooling in ['attention', 'average', 'maximum']
        assert self.img_pooling in ['attention', 'average', 'maximum']
        assert self.head_num in range(1, 13)
        assert self.fusion in ['concatenation', 'attention', 'average', 'maximum']
        self.word_embedding_dim = 128
        self.para_embedding_dim = 768
        self.module_embedding_dim = self.para_embedding_dim
        assert self.module_embedding_dim % self.head_num == 0
    
    def lr(self):
        if self.text_encoder == 'bert' and not (self.remove_name and self.remove_para):
            lr = 2e-5
        elif self.dataset in ['flower'] and not self.remove_img:
            lr = 5e-5
        else:
            lr = 1e-3
        return lr
    
    def batch_size(self, mode, con):
        if self.text_encoder == 'bert':
            batch_size = 8
        else:
            batch_size = 32
        if mode and con:
            batch_size *= 2
        if not mode:
            batch_size *= 4
        return batch_size
    
    def early_stop_time(self):
        early_stop_time = 8
        return early_stop_time
    
    def set_parameters(self, dataset):
        self.label_num = dataset.label_num
        self.img_embedding_dim = len(dataset.imgs[0][-1]) if dataset.imgs[0] != [] else 0
        self.feature_dim = 0
        if not self.remove_name:
            self.feature_dim += self.module_embedding_dim
        if not self.remove_para:
            self.feature_dim += self.module_embedding_dim
        if not self.remove_img:
            self.feature_dim += self.module_embedding_dim
    
    def store_name(self):
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(self.dataset, self.task, self.text_encoder,
        ''.join([str(int(v)) for v in self.remove_info]), ''.join([str(int(v)) for v in self.module_info]),
        self.consistency, self.labeled_num, self.head_num, self.fusion, self.seed)
    
    def parameter_info(self):
        obj = {'dataset': self.dataset, 'task': self.task, 'text_encoder': self.text_encoder,
        'remove_info': self.remove_info, 'module_info': self.module_info,
        'consistency': self.consistency, 'labeled_num': self.labeled_num, 'head_num': self.head_num, 'fusion': self.fusion, 'seed': self.seed}
        return obj