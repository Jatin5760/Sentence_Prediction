# -*- coding:utf-8 -*-

import transformers
import torch
import numpy as np 
import os
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def bert_predict(sentence, target_num, device):
    # sentence: 输入的句子
    # target num: 需要续写的句子数目

    # 读取数据库
    txt_path = './data/txt'
    vec_path = './data/vec'
    files = os.listdir(txt_path)
    sentence_base = []
    vecs = np.zeros((1, 768))

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

    # 载入bert模型
    print('Initialize the bert model...')
    bert_model = transformers.BertModel.from_pretrained('./bert-base-chinese')
    tokenizer = transformers.BertTokenizer.from_pretrained('./bert-base-chinese')

    # 迭代不断来检索下一句
    print('q: ', sentence)
    query = sentence
    for i in range(target_num):
        input_ids = torch.tensor([tokenizer.encode(query)])
        h = bert_model(input_ids)[0][:,0,:].detach().numpy()

        # 根据相似度降序
        score = np.sum(h * vecs, axis=1) / np.linalg.norm(h, axis=1) / np.linalg.norm(vecs, axis=1)
        idx = np.argsort(score)[::-1]

        # 选择检索到的最相似的那句话的下一句
        if sentence_base[idx[0]][0] == query[0]:
            query = sentence_base[idx[1]+1]
        else:
            query = sentence_base[idx[0]+1]
        print(query)


s = "随着各大陆资源的枯竭和环境的恶化，世界把目光投向南极洲。"
bert_predict(s, 10, device)