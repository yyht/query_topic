output_path = '/data/albert.xht/raw_chat_corpus/topic_classification_v4/biake_qa_web_text_zh_train.json.all_risk.v7'

black = []
white = []
import ujson as json
from tqdm import tqdm

with open(output_path) as frobj:
    for line in tqdm(frobj):
        content = json.loads(line.strip())
        if content['score_list']['query_risk'][0][1] > 0.5:
            black.append(content)
        else:
            white.append(content)
            
with open('/data/albert.xht/xiaodao/topic_classification_v7/biake_qa_web_text_zh_train.json.positive', 'w') as fwobj:
    with open(output_path) as frobj:
        for line in tqdm(frobj):
            content = json.loads(line.strip())
            if content['score_list']['query_risk'][0][1] > 0.5:
                if d['score_list']['senti_query'][-1][1] >= 0.6 and d['score_list']['senti'][-1][1] >= 0.6:
                    
                    fwobj.write(json.dumps(d, ensure_ascii=False)+'\n')
            else:
                if content['score_list']['query_risk'][0][1] < 0.2 and d['score_list']['senti_query'][0][1] > 0.6 and d['score_list']['senti'][0][1] > 0.6:
                    