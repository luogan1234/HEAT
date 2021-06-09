import argparse
from data_loader import TypingDataLoader
from processor import Processor
from config import Config
import pickle
import os
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def main():
    if not os.path.exists('result/'):
        os.mkdir('result/')
    if not os.path.exists('result/model_states/'):
        os.mkdir('result/model_states/')
    if not os.path.exists('result/predictions/'):
        os.mkdir('result/predictions/')
    parser = argparse.ArgumentParser(description='Entity-level Typing')
    parser.add_argument('-dataset', type=str, required=True, choices=['meituan_food', 'type_net', 'med_mentions', 'flowers', 'birds'])
    parser.add_argument('-task', type=int, default=0)
    parser.add_argument('-text_encoder', type=str, default='bert', choices=['bert', 'bilstm'])
    parser.add_argument('-remove_name', action='store_true', default=False)
    parser.add_argument('-remove_para', action='store_true', default=False)
    parser.add_argument('-remove_img', action='store_true', default=False)
    parser.add_argument('-without_token_attention', action='store_true', default=False)
    parser.add_argument('-without_cross_modal_attention', action='store_true', default=False)
    parser.add_argument('-consistency', action='store_true', default=False)
    parser.add_argument('-labeled_num', type=int, default=-1)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-cpu', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    assert args.remove_name+args.remove_para+args.remove_img < 3  # should have at least one module
    if args.remove_para:
        args.without_token_attention = True
        args.without_cross_modal_attention = True
    if args.remove_img:
        args.without_cross_modal_attention = True
    remove_info = [args.remove_name, args.remove_para, args.remove_img]
    module_info = [args.without_token_attention, args.without_cross_modal_attention]
    config = Config(args.dataset, args.task, args.text_encoder, remove_info, module_info, args.consistency, args.labeled_num, args.seed, args.cpu)
    model_path = 'result/model_states/{}.pth'.format(config.store_name())
    if os.path.exists(model_path):
        print('experiment done.')
        return
    data_loader = TypingDataLoader(config)
    processor = Processor(data_loader, config)
    processor.train()

if __name__ == '__main__':
    main()