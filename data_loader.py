import numpy as np
import os
import json
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TypingDataset(Dataset):
    def __init__(self, config):
        self.dataset = config.dataset
        self.dataset_path = 'dataset/{}/'.format(self.dataset)
        # name, related paragraphs, label
        with open(os.path.join(self.dataset_path, 'data.pkl'), 'rb') as f:
            self.data = pickle.load(f)
        print('task id: {}'.format(config.task))
        for datum in self.data:
            datum['label'] = datum['label'][config.task]
            datum['paras'] = datum['paras'][:config.para_num]  # only consider the first config.para_num paragraphs
        # cache token ids of entity's name and paragraphs
        file = os.path.join(self.dataset_path, 'data_txt.pkl')
        if os.path.exists(file):
            with open(file, 'rb') as f:
                data_txt = pickle.load(f)
        else:
            print('Start convert name and paragraphs into token ids...')
            if config.language == 'en':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if config.language == 'zh':
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            data_txt = []
            for datum in tqdm.tqdm(self.data):
                name_input = self.convert(datum['name'], config.name_len)
                para_input = [self.convert(para, config.para_len) for para in datum['paras']]
                data_txt.append([name_input, para_input])
            with open(file, 'wb') as f:
                pickle.dump(data_txt, f)
        for i, datum in enumerate(self.data):
            datum['name_input'] = data_txt[i][0]
            datum['para_input'] = data_txt[i][1]
        # calculate label_num
        with open(os.path.join(self.dataset_path, 'types.json'), 'r', encoding='utf-8') as f:
            obj = json.load(f)
            self.types = obj[config.task]
            self.label_num = len(self.types)
        # related images, data_img.pkl consists of a list of img features ([img_num, img_dim] for each entity), and should have the same entity order as self.data
        if not config.remove_img:
            with open(os.path.join(self.dataset_path, 'data_img.pkl'), 'rb') as f:
                self.imgs = pickle.load(f)
        else:
            self.imgs = [[] for i in range(len(self.data))]
        assert len(self.data) == len(self.imgs) > 0
        # splits, train:valid:test=8:1:1
        file = os.path.join(self.dataset_path, 'split.pkl')
        if os.path.exists(file):
            with open(file, 'rb') as f:
                self.split = pickle.load(f)
            for datum in self.data:
                datum['split'] = None
                if datum['name'] in self.split['train']:
                    datum['split'] = 'train'
                if datum['name'] in self.split['valid']:
                    datum['split'] = 'valid'
                if datum['name'] in self.split['test']:
                    datum['split'] = 'test'
        else:
            for datum in self.data:
                datum['split'] = random.sample(['train']*8+['valid', 'test'], 1)[0]
        # limit the number of labeled samples
        if config.labeled_num != -1:
            n = config.labeled_num
            for datum in self.data:
                if datum['split'] == 'train' and datum['label'][0] >= 0:
                    if n > 0:
                        n -= 1
                    else:
                        datum['label'] = [-1]*self.label_num
        print('data handler init finished.')
    
    def convert(self, text, max_len):
        return self.tokenizer.encode(text, truncation=True, max_length=max_len)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        datum = self.data[idx].copy()
        datum['img_input'] = self.imgs[idx]
        return datum

# create batch data for sup and con
class MyCollation:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, data):
        names, name_inputs, para_inputs, img_inputs, labels, para_masks, img_masks = [], [], [], [], [], [], []
        name_len, img_num, para_num, para_len = 0, 0, 0, 0
        for datum in data:
            name_len = max(name_len, len(datum['name_input']))
            img_num = max(img_num, len(datum['img_input']))
            para_num = max(para_num, len(datum['para_input']))
            para_len = max([para_len]+[len(para) for para in datum['para_input']])
        para_num = max(1, min(para_num, self.config.para_num))
        img_num = max(1, min(img_num, self.config.img_num))
        for datum in data:
            names.append(datum['name'])
            # name inputs
            name_input = datum['name_input']
            name_input.extend([0]*(name_len-len(name_input)))
            name_inputs.append(name_input)
            #paragraph inputs
            para_input = []
            for para in datum['para_input']:
                para.extend([0]*(para_len-len(para)))
                para_input.append(para)
            n = len(para_input)
            ids = random.sample(range(n), min(n, para_num))
            para_input = [para_input[id] for id in ids]
            para_mask = [True]*len(para_input)+[False]*(para_num-len(para_input))
            para_masks.append(para_mask)
            for i in range(para_num-len(para_input)):
                para_input.append([0]*para_len)
            para_inputs.append(para_input)
            #image inputs
            img_input = torch.tensor(datum['img_input'], dtype=torch.float)
            n = img_input.size(0)
            ids = random.sample(range(n), min(n, img_num))
            img_input = img_input[ids]
            img_mask = [True]*img_input.size(0)+[False]*(img_num-img_input.size(0))
            img_masks.append(img_mask)
            img_pad = torch.zeros(img_num-img_input.size(0), self.config.img_embedding_dim)
            img_inputs.append(torch.cat([img_input, img_pad], 0))
            # label inputs
            labels.append(datum['label'])
        name_inputs = torch.tensor(name_inputs, dtype=torch.long).to(self.config.device)
        para_inputs = torch.tensor(para_inputs, dtype=torch.long).to(self.config.device)
        img_inputs = torch.stack(img_inputs).to(self.config.device)
        para_masks = torch.tensor(para_masks, dtype=torch.bool).to(self.config.device)
        img_masks = torch.tensor(img_masks, dtype=torch.bool).to(self.config.device)
        res = {'names': names, 'name_inputs': name_inputs, 'para_inputs': para_inputs, 'img_inputs': img_inputs, 'labels': labels, 'para_masks': para_masks, 'img_masks': img_masks}
        return res

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)
        return batch

class TypingDataLoader:
    def __init__(self, config):
        self.dataset = TypingDataset(config)
        config.set_parameters(self.dataset)
        self.config = config
        self.fn = MyCollation(config)
    
    def split(self, data):
        train, valid, test = [], [], []
        for datum in data:
            if datum['split'] == 'train':
                train.append(datum)
            if datum['split'] == 'valid':
                valid.append(datum)
            if datum['split'] == 'test':
                test.append(datum)
        return train, valid, test
    
    def get_train(self):
        train, valid, test = self.split(self.dataset)
        return train, valid, test
    
    def get_predict(self):
        return self.dataset
    
    def filter_data(self, data):
        labeled, unlabeled = [], []
        for datum in data:
            if datum['label'][0] >= 0:
                labeled.append(datum)
            else:
                unlabeled.append(datum)
        return labeled, unlabeled
    
    def create_data_loader(self, data, train, cvt):
        if train:
            data = InfiniteDataLoader(data, self.config.batch_size(train, cvt), shuffle=train, collate_fn=self.fn)
        else:
            data = DataLoader(data, self.config.batch_size(train, cvt), shuffle=train, collate_fn=self.fn)
        return data