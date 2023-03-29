import os, sys

import os, sys
cur_dir_path = os.path.dirname(__file__)
sys.path.extend([cur_dir_path])

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

def build_model(config_path, cur_dir_path):
    import torch, os, sys

    con = configparser.ConfigParser()
    con_path = os.path.join(cur_dir_path, config_path)
    con.read(con_path, encoding='utf8')

    args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
    
    tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)

    from collections import OrderedDict
    schema_dict = OrderedDict({})
    schema2schema_id = {}
    schema_id2schema = {}

    for label_index, schema_info in enumerate(args_path["label_path"].split(',')):
        schema_type, schema_path = schema_info.split(':')
        schema_path = os.path.join(cur_dir_path, schema_path)
        print(schema_type, schema_path, '===schema-path===')
        label2id, id2label = load_label(schema_path)
        schema_dict[schema_type] = {
            'label2id':label2id,
            'id2label':id2label,
            'label_index':label_index
        }
        # print(self.schema_dict[schema_type], '==schema_type==', schema_type)
        schema2schema_id[schema_type] = label_index
        schema_id2schema[label_index] = schema_type

    output_path = os.path.join(cur_dir_path, args_path['output_path'])

    # from roformer import RoFormerModel, RoFormerConfig
    from transformers import BertModel, BertConfig

    config = BertConfig.from_pretrained(args_path["model_path"])
    encoder = BertModel(config=config)

    encoder_net = MyBaseModel(encoder, config)
    
    classifier_list = []

    schema_list = list(schema_dict.keys())

    for schema_key in schema_list:
        classifier = RobertaClassifier(
            hidden_size=config.hidden_size, 
            dropout_prob=con.getfloat('para', 'out_dropout_rate'),
            num_labels=len(schema_dict[schema_key]['label2id']), 
            dropout_type=con.get('para', 'dropout_type'))
        classifier_list.append(classifier)

    classifier_list = nn.ModuleList(classifier_list)

    class MultitaskClassifier(nn.Module):
        def __init__(self, transformer, classifier_list):
            super().__init__()

            self.transformer = transformer
            self.classifier_list = classifier_list

        def forward(self, input_ids=None, input_mask=None, 
                    segment_ids=None, 
                    transformer_mode='cls'):
            hidden_states = self.transformer(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          return_mode=transformer_mode)
            outputs_list = []

            for schema_type, classifier in zip(schema_list, self.classifier_list):

                scores = classifier(hidden_states)
                scores = torch.nn.Softmax(dim=1)(scores)
                outputs_list.append(scores)
            return outputs_list
    net = MultitaskClassifier(encoder_net, classifier_list)
    model_path = os.path.join(output_path, 'multitask_cls.pth.{}'.format(19))
    
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt)
    
    return net, tokenizer, schema_dict

from itertools import chain
def export_onnx(config_path, device, input_names, output_names):
    model, tokenizer, schema_dict = build_model(config_path, cur_dir_path=cur_dir_path)
    con = configparser.ConfigParser()
    con_path = os.path.join(cur_dir_path, config_path)
    con.read(con_path, encoding='utf8')

    args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
    output_path = os.path.join(cur_dir_path, args_path['output_path'])
    
    model.eval()

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        save_path = output_path + "/inference.onnx"
        
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {
                                0: 'batch', 
                                1: 'sequence'
            }
        for name in output_names:
            dynamic_axes[name] = {
                                0: 'batch',
            }

        # dynamic_axes = {name: {0: 'batch', 1: 'sequence'}
        #                 for name in chain(input_names, output_names)}
        
        print(dynamic_axes)

        # Generate dummy input
        batch_size = 2
        seq_length = 6
        dummy_input = []
        for seq_len in [6, 8]:
            dummy_input += [" ".join([tokenizer.unk_token])* seq_len]
        
        inputs = dict(tokenizer(dummy_input, return_tensors="pt", padding=True))
        
        print(inputs)
        
        actual_inputs = {
            'input_ids':inputs['input_ids'].to(device),
            'input_mask':inputs['attention_mask'].to(device),
            'segment_ids':inputs['token_type_ids'].to(device)
        }  
        
        # # print(actual_inputs)
        # actual_inputs = tuple([
        #     inputs['input_ids'].to(device),
        #     inputs['attention_mask'].to(device),
        #     inputs['token_type_ids'].to(device)
        # ])

        torch.onnx.export(model,
                          (actual_inputs,),
                          # actual_inputs,
                          save_path,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=11,
                          verbose=True
                          )

    if not os.path.exists(save_path):
        logger.error(f'Export Failed!')
        
    del model

    return save_path

def validate_onnx(config_path, tokenizer):
    output_path = os.path.join(cur_dir_path, args_path['output_path'])
    save_path = output_path + "/inference.onnx"

    device = torch.device('cpu')

    from onnxruntime import InferenceSession, SessionOptions
    ref_inputs = tokenizer(['你妈是大傻逼'],
                               add_special_tokens=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")

    actual_inputs = {
                'input_ids':ref_inputs['input_ids'],
                'input_mask':ref_inputs['attention_mask'],
                'segment_ids':ref_inputs['token_type_ids']
            }  

    options = SessionOptions()
    session = InferenceSession(str(onnx_path), options, providers=[
                               "CPUExecutionProvider"])

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in actual_inputs.items():
        onnx_inputs[name] = value.numpy()

    onnx_outputs = session.run(None, onnx_inputs)
    print(onnx_outputs)