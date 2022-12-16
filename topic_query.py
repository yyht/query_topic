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

class TopicInfer(object):
	def __init__(self, config_path):

		import torch, os, sys

		con = configparser.ConfigParser()
		con_path = os.path.join(cur_dir_path, config_path)
		con.read(con_path, encoding='utf8')

		args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
		self.tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)

		label_list = []
		label_path = os.path.join(cur_dir_path, args_path['label_path'])
		with open(label_path, 'r') as frobj:
			for line in frobj:
				label_list.append(line.strip())
		n_classes = len(label_list)

		self.label2id, self.id2label = {}, {}
		for idx, label in enumerate(label_list):
			self.label2id[label] = idx
			self.id2label[idx] = label
		
		output_path = os.path.join(cur_dir_path, args_path['output_path'])

		from roformer import RoFormerModel, RoFormerConfig

		config = RoFormerConfig.from_pretrained(args_path["model_path"])
		encoder = RoFormerModel(config=config)
		
		encoder_net = MyBaseModel(encoder, config)

		classify_net = RobertaClassifier(
			hidden_size=config.hidden_size, 
			dropout_prob=con.getfloat('para', 'out_dropout_rate'),
			num_labels=n_classes, 
			dropout_type=con.get('para', 'dropout_type'))

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		class TopicClassifier(nn.Module):
			def __init__(self, transformer, classifier):
				super().__init__()

				self.transformer = transformer
				self.classifier = classifier

			def forward(self, input_ids, input_mask, 
						segment_ids=None, transformer_mode='mean_pooling'):
				hidden_states = self.transformer(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  return_mode=transformer_mode)
				ce_logits = self.classifier(hidden_states)
				return ce_logits, hidden_states

		import os
		self.topic_net = TopicClassifier(encoder_net, classify_net).to(self.device)
		eo = 9
		ckpt = torch.load(os.path.join(output_path, 'cls.pth.{}'.format(eo)), map_location=self.device)
		self.topic_net.load_state_dict(ckpt)
		self.topic_net.eval()

	def predict(self, text, top_n=5):

		"""抽取输入text所包含的类型
		"""
		token2char_span_mapping = self.tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
		encoder_txt = self.tokenizer.encode_plus(text, max_length=256)
		input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(self.device)
		token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(self.device)
		attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(self.device)
		
		with torch.no_grad():
			scores, hidden_states = self.topic_net(input_ids, attention_mask, token_type_ids, 
						 transformer_mode='cls'
						 )
			scores = torch.nn.Softmax(dim=1)(scores)[0].data.cpu().numpy()
		
		schema_types = []
		for index, score in enumerate(scores):
			 schema_types.append([self.id2label[index], float(score)])
		schema_types = sorted(schema_types, key=lambda item:item[1], reverse=True)
		return schema_types[0:5]

topic_api = TopicInfer('./topic_data/config.ini')

text = '王二今天打车去了哪里，从哪里出发，到哪里了'
print(topic_api.predict(text), text)
