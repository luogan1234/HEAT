import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
import os
import json
import numpy as np
import tqdm
import random
import transformers
from model.model import Model
import time
import pickle
import sys
import copy

class Processor(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
        self.sup_loss = self.bce_loss if self.config.loss_fn == 'bce' else self.ce_loss
        self.con_loss = self.con_bce_loss if self.config.loss_fn == 'bce' else self.con_ce_loss
        self.to_label_fn = self.bce_to_label_fn if self.config.loss_fn == 'bce' else self.ce_to_label_fn
    
    def bce_loss(self, outputs, labels):
        labels = torch.tensor(labels, dtype=torch.float).to(self.config.device)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, labels>=0)
        return loss
    
    def ce_loss(self, outputs, labels):
        labels = torch.from_numpy(np.argmax(labels, -1)).to(self.config.device)
        loss = F.cross_entropy(outputs, labels)
        return loss
    
    def con_bce_loss(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        targets_sharpen = (targets/self.config.con_temperature).sigmoid()
        targets = targets.sigmoid()
        masks = (targets > self.config.con_threshold) + ((1-targets) > self.config.con_threshold)
        loss1 = F.mse_loss(outputs, targets_sharpen, reduction='none')
        loss2 = F.mse_loss(1-outputs, 1-targets_sharpen, reduction='none')
        loss = torch.mean(masks*(loss1+loss2))
        return loss
    
    def con_ce_loss(self, outputs, targets):
        outputs = F.softmax(outputs, -1)
        targets_sharpen = F.softmax(targets/self.config.con_temperature, -1)
        targets = F.softmax(targets, -1)
        masks = torch.max(targets, -1)[0] > self.config.con_threshold
        loss = F.mse_loss(outputs, targets_sharpen, reduction='None')
        loss = torch.mean(masks*torch.sum(loss, -1))
        return loss
    
    def bce_to_label_fn(self, outputs):
        return torch.sigmoid(outputs).detach().cpu().numpy()
    
    def ce_to_label_fn(self, outputs):
        return F.softmax(outputs, -1).detach().cpu().numpy()
    
    def dropout_tensor(self, tensor):
        return torch.empty_like(tensor).bernoulli_(1-self.config.con_dropout_rate)*tensor
    
    def dropout_batch(self, batch):
        new_batch = copy.deepcopy(batch)
        # dropout tokens in name and paragraphs
        new_batch['name_inputs'][:, 1:] = self.dropout_tensor(new_batch['name_inputs'][:, 1:])
        new_batch['para_inputs'][:, :, 1:] = self.dropout_tensor(new_batch['para_inputs'][:, :, 1:])
        # dropout whole paragraphs and images
        #new_batch['para_masks'][:, 1:] = self.dropout_tensor(new_batch['para_masks'][:, 1:])
        #new_batch['img_masks'][:, 1:] = self.dropout_tensor(new_batch['img_masks'][:, 1:])
        return new_batch
    
    def train_one_step(self, sup_batch, con_batch, global_steps):
        outputs = self.model(sup_batch)
        loss = self.sup_loss(outputs, sup_batch['labels'])
        sup_loss = loss.item()
        if con_batch is not None:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(con_batch).detach()
            self.model.train()
            dropout_con_batch = self.dropout_batch(con_batch)
            con_outputs = self.model(dropout_con_batch)
            v = self.config.con_weight*min(1, global_steps/self.config.con_warmup)
            loss += v*self.con_loss(con_outputs, outputs)
        con_loss = loss.item()-sup_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), sup_loss, con_loss
    
    def eval_one_step(self, batch):
        with torch.no_grad():
            outputs = self.model(batch)
            loss = self.sup_loss(outputs, batch['labels']).item()
            outputs = self.to_label_fn(outputs)
        return outputs, loss
    
    def evaluate(self, data, flag):
        data = self.data_loader.create_data_loader(data, False, False)
        self.model.eval()
        trues, preds = [], []
        eval_loss = 0
        eval_tqdm = tqdm.tqdm(data, total=len(data))
        eval_tqdm.set_description('eval_loss: {:.4f}'.format(0))
        p1_list = []
        for batch in eval_tqdm:
            outputs, loss = self.eval_one_step(batch)
            for j in range(len(outputs)):
                true = batch['labels'][j]
                pred = outputs[j]
                if true[0] >= 0:
                    trues.append(true)
                    preds.append(pred)
                    p1_list.append(true[np.argmax(outputs[j])])
            eval_loss += loss
            eval_tqdm.set_description('eval_loss: {:.4f}'.format(loss))
        eval_loss /= len(data)
        self.model.train()
        if trues:
            trues, preds = np.array(trues), np.array(preds)
            if self.config.loss_fn == 'bce':
                preds = preds>0.5
            else:
                trues = np.argmax(trues, -1)
                preds = np.argmax(preds, -1)
            acc = accuracy_score(trues, preds)
            p1 = sum(p1_list) / len(p1_list)
            mi_f1 = f1_score(trues, preds, average='micro')
            ma_f1 = f1_score(trues, preds, average='macro')
            print('Average loss for {}: {:.4f}, acc: {:.4f}, p@1: {:.4f}, micro-f1: {:.4f}, macro-f1: {:.4f}.'.format(flag, eval_loss, acc, p1, mi_f1, ma_f1))
        else:
            acc = p1 = mi_f1 = ma_f1 = None
        scores = {'acc': acc, 'p1': p1, 'mi_f1': mi_f1, 'ma_f1': ma_f1}
        return eval_loss, scores
    
    def train(self):
        print('Train starts:')
        self.model = Model(self.config)
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        self.model.to(self.config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr(), eps=1e-8)
        best_para = copy.deepcopy(self.model.state_dict())
        train, valid, test = self.data_loader.get_train()
        train_labeled, train_unlabeled = self.data_loader.filter_data(train)
        valid_labeled, valid_unlabeled = self.data_loader.filter_data(valid)
        test_labeled, test_unlabeled = self.data_loader.filter_data(test)
        print('Labeled data number: train {}, valid {}, test {}.'.format(len(train_labeled), len(valid_labeled), len(test_labeled)))
        print('Unlabeled data number: train {}, valid {}, test {}.'.format(len(train_unlabeled), len(valid_unlabeled), len(test_unlabeled)))
        max_valid_f1, patience, iteration, global_steps = 0.0, 0, 0, 0
        train_sup = self.data_loader.create_data_loader(train_labeled, True, False)
        train_sup_iter = iter(train_sup)
        if self.config.consistency:
            train_con = self.data_loader.create_data_loader(train_unlabeled, True, True)
            train_con_iter = iter(train_con)
        try:
            while patience <= self.config.early_stop_time():
                iteration += 1
                train_loss, train_sup_loss, train_con_loss = 0.0, 0.0, 0.0
                train_tqdm = tqdm.tqdm(range(min(len(train_sup), self.config.max_steps)))
                train_tqdm.set_description('Iteration {} | train_loss: {:.4f}'.format(iteration, 0))
                for steps in train_tqdm:
                    sup_batch = next(train_sup_iter)
                    con_batch = next(train_con_iter) if self.config.consistency else None
                    global_steps += 1
                    loss, sup_loss, con_loss = self.train_one_step(sup_batch, con_batch, global_steps)
                    train_loss += loss
                    train_sup_loss += sup_loss
                    train_con_loss += con_loss
                    train_tqdm.set_description('Iteration {} | train_loss: {:.4f}'.format(iteration, loss))
                steps += 1
                print('Average train_loss: {:.4f}.'.format(train_loss/steps))
                if self.config.consistency:
                    print('Average train_sup_loss: {:.4f}, train_con_loss: {:.4f}.'.format(train_sup_loss/steps, train_con_loss/steps))
                valid_loss, scores = self.evaluate(valid_labeled, 'valid')
                if scores['mi_f1'] > max_valid_f1:
                    patience = 0
                    max_valid_f1 = scores['mi_f1']
                    best_para = copy.deepcopy(self.model.state_dict())
                patience += 1
        except KeyboardInterrupt:
            train_tqdm.close()
            print('Exiting from training early.')
        print('Train finished, max valid f1 {:.4f}, stop at iteration {}.'.format(max_valid_f1, iteration))
        self.model.load_state_dict(best_para)
        test_loss, scores = self.evaluate(test_labeled, 'test')
        print('Test finished, test loss {:.4f}.'.format(test_loss))
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'wb') as f:
            torch.save(best_para, f)
        result_path = 'result/result.txt'
        with open(result_path, 'a', encoding='utf-8') as f:
            obj = self.config.parameter_info()
            obj.update(scores)
            f.write(json.dumps(obj)+'\n')
    
    def predict(self):
        print('Predict starts:')
        self.model = Model(self.config)
        self.model.to(self.config.device)
        print('model parameters number: {}.'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        with open('result/model_states/{}.pth'.format(self.config.store_name()), 'rb') as f:
            best_para = torch.load(f)
        self.model.load_state_dict(best_para)
        self.model.eval()
        data = self.data_loader.get_predict()
        data = self.data_loader.create_data_loader(data, False, False)
        predict_tqdm = tqdm.tqdm(data, total=len(data))
        predict_tqdm.set_description('predict_loss: {:.4f}'.format(0))
        predicts = []
        predict_loss = 0.0
        names = []
        embedding_outs = []
        for batch in predict_tqdm:
            outputs, loss = self.eval_one_step(raw_batch)
            embeddings = self.model.embeddings.detach().cpu()
            for j in range(len(outputs)):
                names.append(batch['names'][j])
                embedding_outs.append(embeddings[j])
                predicts.append({'name': batch['names'][j], 'predict': outputs[j]})
            predict_loss += loss
            predict_tqdm.set_description('predict_loss: {:.4f}'.format(loss))
        print('Average predict_loss: {:.4f}.'.format(predict_loss/len(data)))
        with open('result/predictions/{}.pkl'.format(self.config.store_name()), 'wb') as f:
            pickle.dump(predicts, f)
        with open('result/entity_embedding/{}.pkl'.format(self.config.store_name()), 'wb') as f:
            torch.save({'names': names, 'embeddings': torch.stack(embedding_outs)}, f)