

import torch
import numpy as np
from roformer import RoFormerForCausalLM, RoFormerConfig
from transformers import BertTokenizer
from tqdm import tqdm
import hnswlib
import numpy as np
import pickle
import json

"""
# 可选以下几个。
# junnyu/roformer_chinese_sim_char_small, junnyu/roformer_chinese_sim_char_base
# junnyu/roformer_chinese_sim_char_ft_small, roformer_chinese_sim_char_ft_base
pretrained_model = "junnyu/roformer_chinese_sim_char_base"
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
config = RoFormerConfig.from_pretrained(pretrained_model)
config.is_decoder = True
config.eos_token_id = tokenizer.sep_token_id
config.pooler_activation = "linear"
model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
model.to(device)
model.eval()
# 可选以下几个。
# junnyu/roformer_chinese_sim_char_small, junnyu/roformer_chinese_sim_char_base
# junnyu/roformer_chinese_sim_char_ft_small, roformer_chinese_sim_char_ft_base
pretrained_model = "junnyu/roformer_chinese_sim_char_base"
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
config = RoFormerConfig.from_pretrained(pretrained_model)
config.is_decoder = True
config.eos_token_id = tokenizer.sep_token_id
config.pooler_activation = "linear"
model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
model.to(device)
model.eval()

def gen_synonyms(text, n=100, k=20):
	''''含义： 产生sent的n个相似句，然后返回最相似的k个。
	做法：用seq2seq生成，并用encoder算相似度并排序。
	'''
	# 寻找所有相似的句子
	r = []
	inputs1 = tokenizer(text, return_tensors="pt")
	for _ in range(n):
		inputs1.to(device)
		output = tokenizer.batch_decode(model.generate(**inputs1, top_p=0.95, do_sample=True, max_length=128), skip_special_tokens=True)[0].replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
		r.append(output)
	
	# 对相似的句子进行排序
	r = [i for i in set(r) if i != text and len(i) > 0]
	r = [text] + r
	inputs2 = tokenizer(r, padding=True, return_tensors="pt")
	with torch.no_grad():
		inputs2.to(device)
		outputs = model(**inputs2)
		Z = outputs.pooler_output.cpu().numpy()
	Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
	argsort = np.dot(Z[1:], -Z[0]).argsort()
	
	return [r[i + 1] for i in argsort[:k]]

out = gen_synonyms("广州和深圳哪个好？")
print(out)
"""

class Paraphrase(object):
    def __init__(self, config={}):
        self.config = config
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        """
        self.device = self.config.get('device', 'cpu')

        self.pretrained_model = self.config.get('pretrained_model', 'junnyu/roformer_chinese_sim_char_small')
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        self.model_config = RoFormerConfig.from_pretrained(self.pretrained_model)
        self.model_config.is_decoder = True
        self.model_config.eos_token_id = self.tokenizer.sep_token_id
        self.model_config.pooler_activation = "linear"

        self.model = RoFormerForCausalLM.from_pretrained(self.pretrained_model, config=self.model_config)
        self.model.to(self.device)
        self.model.eval()

    def gen_synonyms(self, text, n=100, k=20):
        ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
        做法：用seq2seq生成，并用encoder算相似度并排序。
        '''
        # 寻找所有相似的句子
        r = []
        inputs1 = self.tokenizer(text, return_tensors="pt")
        for _ in range(n):
            inputs1.to(self.device)
            output = self.tokenizer.batch_decode(self.model.generate(**inputs1, top_p=0.95, do_sample=True, max_length=128), skip_special_tokens=True)[0].replace(" ","").replace(text, "") # 去除空格，去除原始text文本。
            r.append(output)

        # 对相似的句子进行排序
        r = [i for i in set(r) if i != text and len(i) > 0]
        r = [text] + r
        inputs2 = self.tokenizer(r, padding=True, return_tensors="pt")
        with torch.no_grad():
            inputs2.to(self.device)
            outputs = self.model(**inputs2)
            Z = outputs.pooler_output.cpu().numpy()
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        scores = np.dot(Z[1:], Z[0])
        text_score_pairs = [(item, score) for item, score in zip(r[1:], scores)]

        return sorted(text_score_pairs, key=lambda item:item[1], reverse=True)[:k]
    
    def get_embedding(self, text):
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]
        
        inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")
        with torch.no_grad():
            inputs.to(self.device)
            outputs = self.model(**inputs)
            Z = outputs.pooler_output.cpu().numpy()
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        return [(text, feat) for text, feat in zip(text_list, Z)]
    
    def build_index(self, input_path, output_path):
        print(input_path, '==input-path==')
        print(output_path+'/feat.json', '==output-path==')
        
        with open(output_path+'/feat.json', 'w') as fwobj:
            with open(input_path) as frobj:
                queue = []
                t = []
                feat_idx = []
                feat_matrix = []

                for idx, line in tqdm(enumerate(frobj)):
                    content = json.loads(line.strip())
                    queue.append(content['text'])
                    t.append(content)
                    feat_idx.append(idx)

                    if np.mod(len(queue), 32) == 0 and queue:
                        s = self.get_embedding(queue)
                        for result, tt in zip(s, t):
                            tt['feat'] = result[1].tolist()
                            fwobj.write(json.dumps(tt, ensure_ascii=False)+'\n')
                            feat_matrix.append(result[1].tolist())
                        queue = []
                        t = []
                if queue:
                    s = self.get_embedding(queue)
                    for result, tt in zip(s, t):
                        tt['feat'] = result[1].tolist()
                        fwobj.write(json.dumps(tt, ensure_ascii=False)+'\n')
                        feat_matrix.append(result[1].tolist())
                    queue = []
                    t = []
        
        dim = self.model_config.hidden_size
        
        ids = np.array(feat_idx)
        feats = np.array(feat_matrix)

        # Declaring index
        self.p = hnswlib.Index(space = 'ip', dim = dim) # possible options are l2, cosine or ip

        self.p.init_index(max_elements=10000000, ef_construction=10, M=16)

        self.p.set_num_threads(4)

        print("Adding first batch of %d elements" % (feats.shape[0]))
        self.p.add_items(feats, ids, num_threads=4)
        
        self.p.save_index(output_path+'/knn.bin')
        
    def load_knn(self, output_path):
        dim = self.model_config.hidden_size
        self.p = hnswlib.Index(space = 'ip', dim = dim) # possible options are l2, cosine or ip
        self.p.load_index(output_path+'/knn.bin', 
             max_elements=10000000)
        
        with open(output_path+'/feat.json', 'r') as frobj:
            self.meta = {}
            for idx, line in tqdm(enumerate(frobj)):
                content = json.loads(line.strip())
                _del = content.pop('feat')
                self.meta[idx] = content
        
    def get_knn(self, text, topk=10):
        resp_list = self.get_embedding(text)
        result_list = []
        for resp in resp_list:
            labels, distances = self.p.knn_query(resp[1], k = 10)
            distances = (1.0 - distances).tolist()
            output_list = []
            for label, dist in zip(labels.tolist()[0], distances[0]):
                output_list.append({
                    'label':label,
                    'distance':dist,
                    'info':self.meta[label]
                })
                
            result_list.append(output_list)
        return result_list

