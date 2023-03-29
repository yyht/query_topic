
import json, os, sys
from collections import OrderedDict

# def load_topic(data_path, label_path):
#     data_list = []
#     label_list = []
#     label_mapping = {}
#     with open(label_path, 'r') as frobj:
#         for idx, line in enumerate(frobj):
#             label_list.append(line.strip())
#             label = line.strip()
#             label_mapping[label] = idx
            
#     with open(data_path, 'r') as frobj:
#         for idx, line in enumerate(frobj):
#             content = json.loads(line.strip())
#             d = {
#                 'text_a':content['text'],
#                 'text_b':'',
#                 "question": "下面文本属于哪一个类别？",
#                 "choice": label_list,
#                 "answer": content['label'][0], 
#                 "label": label_mapping[content['label'][0]], 
#                 "id": idx,
#                 'label_mapping':label_mapping
#             }
#             data_list.append(d)
#     return data_list

# def load_senti(data_path, label_path):
#     data_list = []
#     label_list = []
#     label_mapping = {}
#     with open(label_path, 'r') as frobj:
#         for idx, line in enumerate(frobj):
#             label_list.append(line.strip())
#             label = line.strip()
#             label_mapping[label] = idx
    
#     label_str_mapping = OrderedDict({})
#     for label in label_list:
#         label_str_mapping[label] = '这个句子的情感是{}'.format(label)
            
#     with open(data_path, 'r') as frobj:
#         for idx, line in enumerate(frobj):
#             content = json.loads(line.strip())
            
#             choice = [label_str_mapping[key] for key in label_str_mapping]
            
#             d = {
#                 'text_a':content['text'],
#                 'text_b':'',
#                 "question": "",
#                 "choice": label_list,
#                 "answer": content['label'][0], 
#                 "label": label_mapping[content['label'][0]], 
#                 "id": idx,
#                 'label_mapping':label_mapping
#             }
#             data_list.append(d)
#     return data_list
    
def load_hhrlfh(data_path, label_path, split='train'):
    data_list = []
    label_list = []
    label_mapping = {}
    with open(label_path, 'r') as frobj:
        for idx, line in enumerate(frobj):
            label_list.append(line.strip())
            label = line.strip()
            label_mapping[label] = idx
            
    label_str_mapping = OrderedDict({})
    for label in label_list:
        label_str_mapping[label] = '这是一个被{}的回复'.format(label)
        
    with open(data_path, 'r') as frobj:
        for idx, line in enumerate(frobj):
            content = json.loads(line.strip())
            if content.get('source', '') != split:
                continue
                
            if 'human_translate' not in content or 'assistant_translate' not in content:
                continue
                
            choice = [label_str_mapping[key] for key in label_str_mapping]
                
            d = {
                'texta':content['human_translate'],
                'textb':content['assistant_translate'],
                "question": '',   
                "choice": choice,
                "answer": content['label'], 
                "label": label_mapping[content['label']], 
                "id": idx,
                'label_list':label_list
            }
            data_list.append(d)
    import random
    print(data_list[0], '==data list==')
    random.shuffle(data_list)
    return data_list    
            
