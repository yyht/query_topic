
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

# cur_dir_path = '/root/xiaoda/query_topic/'

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

class TorchInfer(object):
    def __init__(self, config_path):

        import torch, os, sys
        from nets.them_classifier import MyBaseModel, RobertaClassifier
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import BertTokenizerFast

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, config_path)
        con.read(con_path, encoding='utf8')

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        self.tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)

        from collections import OrderedDict
        self.schema_dict = OrderedDict({})
        self.schema2schema_id = {}
        self.schema_id2schema = {}

        for label_index, schema_info in enumerate(args_path["label_path"].split(',')):
            schema_type, schema_path = schema_info.split(':')
            schema_path = os.path.join(cur_dir_path, schema_path)
            print(schema_type, schema_path, '===schema-path===')
            label2id, id2label = load_label(schema_path)
            self.schema_dict[schema_type] = {
                'label2id':label2id,
                'id2label':id2label,
                'label_index':label_index
            }
            # print(self.schema_dict[schema_type], '==schema_type==', schema_type)
            self.schema2schema_id[schema_type] = label_index
            self.schema_id2schema[label_index] = schema_type
        
        output_path = os.path.join(cur_dir_path, args_path['output_path'])

        # from roformer import RoFormerModel, RoFormerConfig
        from transformers import BertModel, BertConfig

        config = BertConfig.from_pretrained(args_path["model_path"])
        encoder = BertModel(config=config)
        
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
                        dt_idx=None, mode='predict'):
                hidden_states = self.transformer(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              return_mode=transformer_mode)
                outputs_list = []
                
                for idx, classifier in enumerate(self.classifier_list):
                    
                    if dt_idx:
                        if idx not in dt_idx:
                            outputs_list.append([])
                            continue
                    
                    scores = classifier(hidden_states)
                    if mode == 'predict':
                        scores = torch.nn.Softmax(dim=1)(scores)
                    outputs_list.append(scores)
                return outputs_list, hidden_states

        self.net = MultitaskClassifier(encoder_net, classifier_list).to(self.device)
        
    def reload(self, model_path):
        import torch
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt)
        self.net.eval()
        if self.device != 'cpu':
            self.net = self.net.half()

    def predict(self, text, allowed_schema_type={}):
        """抽取输入text所包含的类型
        """
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]
        model_input = self.tokenizer(text_list, return_tensors="pt",padding=True)
        for key in model_input:
            model_input[key] = model_input[key].to(self.device)
        
        allowed_schema_type_ids = {}
        for schema_type in allowed_schema_type:
            allowed_schema_type_ids[self.schema2schema_id[schema_type]] = schema_type

        with torch.no_grad():
            [logits_list, 
            hidden_states] = self.net(model_input['input_ids'], 
                model_input['attention_mask'], 
                model_input['token_type_ids'], transformer_mode='cls', dt_idx=allowed_schema_type_ids)
            
        score_dict_list = self.postprocess(text_list, logits_list, allowed_schema_type)
        return score_dict_list
    
    def postprocess(self, text_list, logits_list, allowed_schema_type):
        score_dict_list = []
        for idx, text in enumerate(text_list):
            scores_dict = {}
            for schema_idx, (schema_type, scores) in enumerate(zip(list(self.schema_dict.keys()), logits_list)):
                if allowed_schema_type:
                    if schema_type not in allowed_schema_type:
                        continue
                # scores = torch.nn.Softmax(dim=1)(logits)[idx].data.cpu().numpy()
                scores = scores[idx].data.cpu().numpy()
                scores_dict[schema_type] = []
                for index, score in enumerate(scores):
                    scores_dict[schema_type].append([self.schema_dict[schema_type]['id2label'][index], 
                                            float(score)])
                if schema_type in ['topic']:
                    schema_type_scores = sorted(scores_dict[schema_type], key=lambda item:item[1], reverse=True)
                    scores_dict[schema_type] = schema_type_scores[0:5]
            score_dict_list.append(scores_dict)
        return score_dict_list
    
class OnnxInfer(object):
    def __init__(self, config_path, use_fp16=False):
        import os, sys
        from onnxruntime import InferenceSession, SessionOptions
        from transformers import BertTokenizerFast

        con = configparser.ConfigParser()
        con_path = os.path.join(cur_dir_path, config_path)
        con.read(con_path, encoding='utf8')

        args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
        self.tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)

        from collections import OrderedDict
        self.schema_dict = OrderedDict({})
        self.schema2schema_id = {}
        self.schema_id2schema = {}

        for label_index, schema_info in enumerate(args_path["label_path"].split(',')):
            schema_type, schema_path = schema_info.split(':')
            schema_path = os.path.join(cur_dir_path, schema_path)
            print(schema_type, schema_path, '===schema-path===')
            label2id, id2label = load_label(schema_path)
            self.schema_dict[schema_type] = {
                'label2id':label2id,
                'id2label':id2label,
                'label_index':label_index
            }
            # print(self.schema_dict[schema_type], '==schema_type==', schema_type)
            self.schema2schema_id[schema_type] = label_index
            self.schema_id2schema[label_index] = schema_type
                    
        self.output_path = os.path.join(cur_dir_path, args_path['output_path'])
        
        from onnxruntime import InferenceSession, SessionOptions
        onnx_model = float_onnx_file = os.path.join(
            self.output_path, "inference.onnx")
        
        if os.path.exists(os.path.join(self.output_path, "multitask_cls.pth.19")) and not os.path.exists(os.path.join(self.output_path, "inference.onnx")):
            from export_onnx_model import export_onnx
            input_names = [
                'input_ids',
                'input_mask',
                'segment_ids',
            ]
            output_names = list(self.schema_dict.keys())

            save_path = export_onnx(config_path, 'cpu', input_names, output_names)
            print(save_path)
            
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
            
        if self.device == "cpu":
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = ['CUDAExecutionProvider']
            print(self.providers, '==providers==')
            if use_fp16:
                from onnxconverter_common import float16
                import onnx
                fp16_model_file = os.path.join(self.output_path,
                                               "fp16_inference.onnx")
                onnx_model = onnx.load_model(float_onnx_file)
                trans_model = float16.convert_float_to_float16(
                    onnx_model, keep_io_types=True)
                onnx.save_model(trans_model, fp16_model_file)
                onnx_model = fp16_model_file
                
        self.sess_options = SessionOptions()
        print(self.providers, '==providers==')
        self.sess_options.intra_op_num_threads = 8
        self.predictor = InferenceSession(
            onnx_model, sess_options=self.sess_options, providers=self.providers)
        
        print(self.predictor.get_providers())
        
    def predict(self, text, allowed_schema_type={}):
        """抽取输入text所包含的类型
        """
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]
        model_input = self.tokenizer(text_list, padding=True, max_length=256, truncation=True) 
        
        actual_inputs = {
            'input_ids':model_input['input_ids'],
            'input_mask':model_input['attention_mask'],
            'segment_ids':model_input['token_type_ids']
        }  
        
        onnx_inputs = {}
        for name, value in actual_inputs.items():
            onnx_inputs[name] = value
        
        onnx_outputs = self.predictor.run(None, onnx_inputs)
        score_dict_list = self.postprocess(text_list, onnx_outputs, allowed_schema_type)
        return score_dict_list
    
    def postprocess(self, text_list, logits_list, allowed_schema_type):
        score_dict_list = []
        for idx, text in enumerate(text_list):
            scores_dict = {}
            for schema_idx, (schema_type, scores) in enumerate(zip(list(self.schema_dict.keys()), logits_list)):
                if allowed_schema_type:
                    if schema_type not in allowed_schema_type:
                        continue
                # scores = torch.nn.Softmax(dim=1)(logits)[idx].data.cpu().numpy()
                scores = scores[idx]
                scores_dict[schema_type] = []
                for index, score in enumerate(scores):
                    scores_dict[schema_type].append([self.schema_dict[schema_type]['id2label'][index], 
                                            float(score)])
                if schema_type in ['topic', 'cmid']:
                    schema_type_scores = sorted(scores_dict[schema_type], key=lambda item:item[1], reverse=True)
                    scores_dict[schema_type] = schema_type_scores[0:5]
            score_dict_list.append(scores_dict)
        return score_dict_list
    
topic_risk_api = OnnxInfer('./risk_data_tiny/config_topic_risk_green_v1.ini', use_fp16=False)
# topic_risk_api = OnnxInfer('./risk_data_tiny_cmid/risk_data_tiny/config_topic_risk_green_v1.ini', use_fp16=False)

text = '木耳炒鸡蛋'
import time
start = time.time()
itera = 1000
for i in range(itera):
    resp = topic_risk_api.predict([text]*1)
print((time.time()-start)/itera)
print(resp[0])
        
        
        

