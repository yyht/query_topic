

import json
import sys
import numpy as np
from datetime import timedelta

import os, sys
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

import configparser
from tqdm import tqdm
import torch

class UniMCInfer(object):
    def __init__(self, config_path):

        import torch, os, sys
        from nets.unimc import UniMCModel, UniMCDataset
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import BertTokenizerFast

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, config_path)
        con.read(con_path, encoding='utf8')

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))

        if args_path['model_type'] == 'albert':
            from transformers import AlbertTokenizerFast
            tokenizer = AlbertTokenizerFast.from_pretrained(
                        args_path["model_path"])
        else:
            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast.from_pretrained(
                        args_path["model_path"])

        if args_path['language'] == 'chinese':
            yes_token = tokenizer.encode('是')[1]
            no_token = tokenizer.encode('非')[1]
        else:
            yes_token = tokenizer.encode('yes')[1]
            no_token = tokenizer.encode('no')[1]
        
        output_path = os.path.join(cur_dir_path, args_path['output_path'])

        dataset = UniMCDataset([], yes_token, no_token, tokenizer, con, used_mask=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = UniMCModel(args_path["model_path"], yes_token)
        
    def reload(self, model_path):
        import torch
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt)
        self.net.eval()
        if self.device != 'cpu':
            self.net = self.net.half()
        self.collate_fn = self.dataset.collate_fn

    import copy
    def predict(self, batch_data, dataset, device):
        batch = [dataset.encode(
                sample) for sample in batch_data]
        batch = self.collate_fn(batch)
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            _, _, logits = self.net(**batch)
        soft_logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = torch.argmax(soft_logits, dim=-1).detach().cpu().numpy()
        soft_logits = soft_logits.detach().cpu().numpy()
        clslabels_mask = batch['clslabels_mask'].detach(
            ).cpu().numpy().tolist()
        clslabels = batch['clslabels'].detach().cpu().numpy().tolist()
        for i, v in enumerate(batch_data):
            label_idx = [idx for idx, v in enumerate(
                clslabels_mask[i]) if v == 0.]
            label = label_idx.index(logits[i])
            answer = batch_data[i]['choice'][label]
            score = {}
            for c in range(len(batch_data[i]['choice'])):
                score[batch_data[i]['choice'][c]] = float(
                    soft_logits[i][label_idx[c]])

            batch_data[i]['label_ori'] = copy.deepcopy(batch_data[i]['label'])
            batch_data[i]['label'] = label
            batch_data[i]['answer'] = answer
            batch_data[i]['score'] = score

        return batch_data