# -*- coding:utf-8 -*-
import json
import os

# json 文件的地址
data_path = './data/original'
store_path = './data/data'

files = os.listdir(data_path)
for file in files:
    sentence_list = []
    refine_list = []
    file_path = os.path.join(data_path, file)
    with open(file_path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
        sentence_list = load_dict['content']
        new_sentence_split = []
        refine_list = []
        for sentence in sentence_list:
            sentence_split = sentence.split('。')[:-1]
            for s in sentence_split:
                s = s+'。'
                if len(s) < 10:
                    continue
                refine_list.append(s)
        for i in range(len(refine_list)-1):
            if refine_list[i+1][0]=='”':
                new_sentence_split.append(refine_list[i]+'”')
                refine_list[i+1] = refine_list[i+1][1:]
            else:
                new_sentence_split.append(refine_list[i])
        for i in range(len(new_sentence_split)):
            if "“" in new_sentence_split[i] and "”" not in new_sentence_split[i]:
                new_sentence_split[i] = new_sentence_split[i] + "”"
        load_dict['content'] = new_sentence_split
    with open(os.path.join(store_path, file), 'w', encoding='utf-8') as f:
        json.dump(load_dict, f, ensure_ascii=False, indent=4)


