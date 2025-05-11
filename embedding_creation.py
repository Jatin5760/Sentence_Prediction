# -*- coding:utf-8 -*-

import transformers
from data import sentence_embedding
import os
import numpy as np
# 中文bert初始化
bert_model = transformers.BertModel.from_pretrained('./bert-base-chinese')
tokenizer = transformers.BertTokenizer.from_pretrained('./bert-base-chinese')

# json文件存放地址: ./data/data, 生成结果也会存放在对应的目录下面 ./data/txt, ./data/vec
path = './data'

# 将向量文件归并到一个文件，方便训练。
def merger_vec(path='./data'):
    vec_path = os.path.join(path, 'vec')
    files = os.listdir(vec_path)
    vectors = np.zeros((1, 768))
    for file in files:
        print(file)
        try:
            mat = np.loadtxt(os.path.join(vec_path, file))
            vectors = np.concatenate((vectors, mat), axis=0)
        except:
            print('failed:', file)
    vectors = vectors[1:]
    np.savetxt(os.path.join(path, 'embedding_vector.txt'), vectors, fmt='%f')

print('begin embedding...')
# 使用bert将句子进行编码
sentence_embedding(bert_model, tokenizer, path)

# 将向量文件归并到一个文件，方便模型的训练。
# merger_vec('./data')
print('merge success!')


