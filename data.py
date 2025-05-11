# -*- coding:utf-8 -*-
import torch
import os
import json
import numpy as np
import transformers

def sentence_embedding(model, tokenizer, path='./data'):
    txt_path = os.path.join(path, 'data')
    res_path = os.path.join(path, 'txt')
    vec_path = os.path.join(path, 'vec')
    files = os.listdir(txt_path)
    # 读取当前目录下面所有文件的句子。
    print('begin to read data!')

    for file in files:
        print(file)
        file_path = os.path.join(txt_path, file)
        with open(file_path, 'r', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            sentence_list = load_dict['content']

            vector_list = []
            # 去掉长度>256 或者 < 10的句子
            new_sentence_list = []
            # print('processing data...')
            for sentence in sentence_list:
                if len(sentence) > 256 or len(sentence) < 10:
                    continue
            # print(sentence)
                try:
                    input_ids = torch.tensor([tokenizer.encode(sentence)])
                    h = model(input_ids)[0][:,0,:].detach().numpy()
                except:
                    print(sentence)
                new_sentence_list.append(sentence)
                vector_list += [(h)]
            print('hidden states size: ', len(vector_list))
            print('total sentence number:', len(new_sentence_list))
            
            # print(vector_list.shape)

            # 句子和向量都保存到对应文件中。
            np_vector_list = np.array(vector_list).reshape((-1, 768))
            np.savetxt(os.path.join(vec_path, file), np_vector_list, fmt='%f')
            sentence_dict = {'content': []}
            sentence_dict['content'] = new_sentence_list
            with open(os.path.join(res_path, file), 'w', encoding='utf-8') as f:
                json.dump(sentence_dict, f, ensure_ascii=False, indent=4)

def batchify(data, length_num, batch_size):
    inp = torch.rand(1, 768*length_num).float()
    target = torch.rand(1, 768).float()
    for i in range(batch_size):
        inp_t = torch.from_numpy(data[i:i+length_num,:]).float().view(1, -1)
        inp = torch.cat((inp, inp_t), 0)
        target_t = torch.from_numpy(data[i+length_num,:]).float().view(1, -1)
        target = torch.cat((target, target_t), 0)
    return inp[1:, :], target[1:, :]

class DataLoader(object):
    def __init__(self, batch_size=2, length_num=4, path='./data'):
        # length_num 需要前面 n 句话来预测下一句的内容
        self.batch_size = batch_size
        file_path = os.path.join(path, 'embedding_vector.txt')
        self.data = np.loadtxt(file_path)
        print(self.data.shape)
        self.epoch = 0
        self.length_num = length_num

    def __iter__(self):
        idx = 0
        self.epoch += 1
        while idx+self.batch_size+self.length_num+1 < len(self.data):
            yield batchify(self.data[idx: idx+self.batch_size+self.length_num], self.length_num, self.batch_size)
            idx += self.batch_size


