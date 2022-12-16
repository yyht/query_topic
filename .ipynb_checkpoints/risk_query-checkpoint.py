import torch
import json
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast
import transformers
from datetime import timedelta

import os, sys
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

from nets.them_classifier import MyBaseModel, RobertaClassifier

import configparser
from tqdm import tqdm

def load_label(filepath):
	label_list = []
	with open(filepath, 'r') as frobj:
		for line in frobj:
			label_list.append(line.strip())
		n_classes = len(label_list)

		label2id = {}
		id2label = {}
		for idx, label in enumerate(label_list):
			label2id[label] = idx
			id2label[idx] = label
		return label2id, id2label

class RiskInfer(object):
	def __init__(self, config_path):

		import torch, os, sys

		con = configparser.ConfigParser()
		con_path = os.path.join(cur_dir_path, config_path)
		con.read(con_path, encoding='utf8')

		args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
		self.tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)

		from collections import OrderedDict
		self.schema_dict = OrderedDict({})

		for label_index, schema_info in enumerate(args_path["label_path"].split(',')):
			schema_type, schema_path = schema_info.split(':')
			print(schema_type, schema_path, '===schema-path===')
			label2id, id2label = load_label(schema_path)
			self.schema_dict[schema_type] = {
				'label2id':label2id,
				'id2label':id2label,
				'label_index':label_index
			}
			print(self.schema_dict[schema_type], '==schema_type==', schema_type)
		
		output_path = os.path.join(cur_dir_path, args_path['output_path'])

		from roformer import RoFormerModel, RoFormerConfig

		config = RoFormerConfig.from_pretrained(args_path["model_path"])
		encoder = RoFormerModel(config=config)
		
		encoder_net = MyBaseModel(encoder, config)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		classifier_list = []

		schema_list = list(self.schema_dict.keys())

		for schema_key in schema_list:
			classifier = RobertaClassifier(
				hidden_size=config.hidden_size, 
				dropout_prob=con.getfloat('para', 'out_dropout_rate'),
				num_labels=len(self.schema_dict[schema_key]['label2id']), 
				dropout_type=con.get('para', 'dropout_type'))
			classifier_list.append(classifier)

		classifier_list = nn.ModuleList(classifier_list)

		class MultitaskClassifier(nn.Module):
			def __init__(self, transformer, classifier_list):
				super().__init__()

				self.transformer = transformer
				self.classifier_list = classifier_list

			def forward(self, input_ids, input_mask, 
						segment_ids=None, 
						transformer_mode='mean_pooling', 
						dt_idx=None):
				hidden_states = self.transformer(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  return_mode=transformer_mode)
				outputs_list = []
				
				for idx, classifier in enumerate(self.classifier_list):
					
					if dt_idx is not None and idx != dt_idx:
						continue
					
					ce_logits = classifier(hidden_states)
					outputs_list.append(ce_logits)
				return outputs_list, hidden_states

		self.net = MultitaskClassifier(encoder_net, classifier_list).to(self.device)

		eo = 9
		ckpt = torch.load(os.path.join(output_path, 'multitask_cls.pth.{}'.format(eo)), map_location=self.device)
		self.net.load_state_dict(ckpt)
		self.net.eval()

	def predict(self, text):

		"""抽取输入text所包含的类型
		"""
		encoder_txt = self.tokenizer.encode_plus(text, max_length=256)
		input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(self.device)
		token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(self.device)
		attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(self.device)
		
		scores_dict = {}
		with torch.no_grad():
			[logits_list, 
			hidden_states] = self.net(input_ids, 
				attention_mask, token_type_ids, transformer_mode='cls')
		for schema_type, logits in zip(list(self.schema_dict.keys()), logits_list):
			scores = torch.nn.Softmax(dim=1)(logits)[0].data.cpu().numpy()
			scores_dict[schema_type] = []
			for index, score in enumerate(scores):
				scores_dict[schema_type].append([self.schema_dict[schema_type]['id2label'][index], 
										float(score)])
		return scores_dict

risk_api = RiskInfer('./risk_data/config.ini')

text = '跟心灵鸡汤没什么本质区别嘛，至少我不喜欢这样读经典，把经典都解读成这样有点去中国化的味道了'
print(risk_api.predict(text), text)
