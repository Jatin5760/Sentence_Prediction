# -*- coding:utf-8 -*-

import transformers
import torch
import numpy as np 
import os
import json
import time
import faiss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# 读取数据库
txt_path = './data/txt'
vec_path = './data/vec'
files = os.listdir(txt_path)
sentence_base = []
vecs = np.zeros((1, 768))

database_time = time.time()
print('begin to read database!')
for file in files:
    file_path = os.path.join(txt_path, file)
    with open(file_path, 'r', encoding='utf-8')  as load_f:
        load_dict = json.load(load_f)
        t_sentences = load_dict['content']
        sentence_base += t_sentences
    v_path = os.path.join(vec_path, file)
    mat = np.loadtxt(v_path)
    mat = mat.reshape(-1, 768)
    vecs = np.concatenate((vecs, mat), axis=0)
vecs = vecs[1:]
print('all data base has been read.')
print('sentence number: ', len(vecs), len(sentence_base))
database_time_end = time.time()
print('time consume:', database_time_end - database_time)

# 建立faiss
d = 768
index = faiss.IndexFlatL2(d)
print(index.is_trained)
#正则化之后加入索引
vecs = vecs.astype('float32')
#vecs = vecs / np.linalg.norm(vecs, axis=1)
index.add(vecs)
print(index.ntotal)

# 载入bert模型
print('Initialize the bert model...')
bert_model = transformers.BertModel.from_pretrained('./bert-base-chinese')
tokenizer = transformers.BertTokenizer.from_pretrained('./bert-base-chinese')

def bert_predict(sentence, target_num, device):
    # sentence: 输入的句子
    # target num: 需要续写的句子数目

    # 迭代不断来检索下一句
    # print('q: ', sentence)
    query = sentence
    sentence_his = []
    sentence_his.append(query)
    for i in range(target_num):
        input_ids = torch.tensor([tokenizer.encode(query)])
        h = bert_model(input_ids)[0][:,0,:].detach().numpy().astype('float32')

        D, I = index.search(h.reshape(1, -1), 1)
        print('query:', query)
        print('most similar:', sentence_base[I[0][0]])
        print('next sentence:', sentence_base[I[0][0]+1])

'''
        # 根据相似度降序
        score = np.sum(h * vecs, axis=1) / np.linalg.norm(h, axis=1) / np.linalg.norm(vecs, axis=1)
        idx = np.argsort(score)[::-1]

        # 选择检索到的最相似的那句话的下一句
        print('query:', query)
        j = 0
        while sentence_base[idx[j]] in sentence_his:
            j+=1
        print('most similar:', sentence_base[idx[j]])
        print('next sentence:', sentence_base[idx[j]+1])
        query = query + sentence_base[idx[j]+1]
        sentence_his.append(sentence_base[idx[j]])
        sentence_his.append(sentence_base[idx[j]+1])
'''

start_time = time.time()
s = "2100年，地球被火星入侵"
bert_predict(s, 1, device)
s1 = "一场大水仿佛淹没了整个城市"
bert_predict(s1, 1, device)
s2 = "生而为人，为什么如此艰难"
bert_predict(s2, 1, device)
s3 = "公元3000年，人类已经踏上了对宇宙扩张的过程。"
bert_predict(s3, 1, device)
s4 = "人类文明看似繁花似锦，我们可以凿穿高山，可以填海造陆，可以控制降雨……我们人类文明看似很强大。"
end_time = time.time()
print('total time: ', end_time - start_time)