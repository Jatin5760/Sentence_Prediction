# -*- coding:utf-8 -*-
import numpy as np
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

def beam_search(sentence_list, device, target_num=10, beam_size=5):
    # target_num 想要连续生成的句子数

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

    # 预测模型载入
    model_path = './model/CKPT'
    residual_model = model_init(model_path, device, length_num=4)
    print('predict model loaded.')
    
    # beam search, 存储格式为[[sequence_list], socre]
    beam_candidates = [[sentence_list, 1.0]]
    for n in range(target_num):
        # 存储所有的候选 大小为 k * k
        all_candidates = list()
        #print(len(beam_candidates))
        for i in range(len(beam_candidates)):
            sequence, score = beam_candidates[i]
            # 取最新的4句话用于检索下一句
            seq = sequence[-4:]
            #print(seq)
            # 将4句话encode成为向量
            h = torch.tensor([[1.]])
            for sentence in sentence_list:
                input_ids = torch.tensor([tokenizer.encode(sentence)])
                h_t = bert_model(input_ids)[0][:,0,:]
                h = torch.cat((h, h_t), dim=1)
            h = h[0, 1:].view(1, -1).to(device)
            # 模型做出预测
            res = residual_model.work(h)
            res_vec = res.cpu().detach().numpy()

            scores = np.sum(res_vec * vecs, axis=1) / np.linalg.norm(res_vec, axis=1) / np.linalg.norm(vecs, axis=1)
            idx = np.argsort(scores)[::-1]
            # 将得分最高的 k 句话存入
            for j in range(beam_size):
                #print(sentence_base[idx[j]])
                candidate = [sequence+ [sentence_base[idx[j]]], score * scores[idx[j]]]
                all_candidates.append(candidate)

            # 所有候选值 k*k 进行排序，选择前 k 个
        #print(len(all_candidates))
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        beam_candidates = ordered[:beam_size]
    return beam_candidates


s=["随着各大陆资源的枯竭和环境的恶化，世界把目光投向南极洲。",
   "南极突然崛起的两大强国在世界政治格局中取得了与他们在足球场上同样的地位，使得南极条约成为一纸空文。",
   "但人类的理智在另一方面取得了胜利，全球彻底销毁核武器的最后进程开始了。",
   "随着全球无核化的实现，人类对南极大陆的争夺变得安全了一些。",]
res = beam_search(s, device, target_num=5, beam_size=3)
print(res)