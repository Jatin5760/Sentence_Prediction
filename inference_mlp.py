# -*- coding:utf-8 -*-
import transformers
import torch
from model import ResidualModel
import numpy as np
import json
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def model_init(model_path, device, length_num):
    ckpt = torch.load(model_path, map_location='cpu')
    residual_model = ResidualModel(device=device, inp_size=length_num*768, output_size=768,\
    num_residual_layers=2, dropout=0.5)
    residual_model.load_state_dict(ckpt['model'])
    residual_model = residual_model.to(device)
    residual_model.eval()
    return residual_model



# greedy search
def sentence_predict(sentence_list, length_num, target_num, device):
    # sentence_list (bsz, 4)
    # length_num: 根据前面length_num句话来预测下一句
    # target_num: 准备连续生成的句子数目

    # 初始化bert
    print('Initialize the bert model...')
    bert_model = transformers.BertModel.from_pretrained('./bert-base-chinese')
    tokenizer = transformers.BertTokenizer.from_pretrained('./bert-base-chinese')

    # 读取文字和向量数据库
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

    # 预测模型载入
    model_path = './model/CKPT'
    residual_model = model_init(model_path, device, length_num)
    print('predict model loaded.')

    print('pre-content:', sentence_list)
    h = torch.tensor([[1.]])
    for sentence in sentence_list:
        input_ids = torch.tensor([tokenizer.encode(sentence)])
        h_t = bert_model(input_ids)[0][:,0,:]
        h = torch.cat((h, h_t), dim=1)
    h = h[0, 1:].view(1, -1).to(device)
    #print(h.size())
    res = residual_model.work(h)
    res_vec = res.cpu().detach().numpy()
    #print(res_vec.shape)

    score = np.sum(res_vec * vecs, axis=1) / np.linalg.norm(res_vec, axis=1) / np.linalg.norm(vecs, axis=1)
    idx = np.argsort(score)[::-1]
    res_sequence = sentence_base[idx[0]]

    print(res_sequence)
    for i in range(target_num-1):
        # 将新生成的[1,768]拼接起来
        h = torch.cat((h, res), dim=1)
        # 去点前面的部分
        h = h[0, 768:].view(1, -1).to(device)
        res = residual_model.work(h)
        res_vec = res.cpu().detach().numpy()
        score = np.sum(res_vec * vecs, axis=1) / np.linalg.norm(res_vec, axis=1) / np.linalg.norm(vecs, axis=1)
        idx = np.argsort(score)[::-1]
        if sentence_base[idx[0]][0] == res_sequence[0]:
            res_sequence = sentence_base[idx[1]]
        else:
            res_sequence = sentence_base[idx[0]]
        print(res_sequence)






s=["随着各大陆资源的枯竭和环境的恶化，世界把目光投向南极洲。",
   "南极突然崛起的两大强国在世界政治格局中取得了与他们在足球场上同样的地位，使得南极条约成为一纸空文。",
   "但人类的理智在另一方面取得了胜利，全球彻底销毁核武器的最后进程开始了，随着全球无核化的实现，人类对南极大陆的争夺变得安全了一些。",
   "全球核武器的最后销毁采用两种方式：拆卸和地下核爆炸。这是位于中国的地下爆炸销毁点之一。",]
sentence_predict(s, 4, 10, device)
