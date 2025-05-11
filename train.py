# -*- coding:utf-8 -*-

from data import sentence_embedding, DataLoader
import transformers
from model import ResidualModel
import torch
from optim import Optim
import numpy as np 
import random

length_num = 4 # 每次根据前面Num句话来预测下一句
data_path = './data' # 存放数据的地址
model_path = './model' # 存放ckpt的地址
batch_size = 128
print_every = 100
save_every =3000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 残差MLP模型初始化
residual_model = ResidualModel(device=device, inp_size=length_num*768, output_size=768,\
    num_residual_layers=2, dropout=0.5)
residual_model = residual_model.to(device)

# 优化器初始化
optimizer = Optim(model_size=1024, factor=1, warmup=10000,\
    optimizer=torch.optim.Adam(residual_model.parameters(), lr=1.0, betas=(0.9, 0.998), eps=1e-9)) 

# 读取预处理完成的数据集合
train_data = DataLoader(batch_size=batch_size, length_num=length_num, path=data_path)

# 设置随机种子
def setup_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seeds(2020)
batch_acm = 0
acc_acm, loss_acm =0., 0.


while True:
    residual_model.train()
    for inp, target in train_data:
    # inp=(batch_size, length_number*768), target=(batch_size, 768)
        batch_acm += 1

        inp = inp.to(device)
        target = target.to(device)

        residual_model.zero_grad()
        res, loss, acc = residual_model(inp, target)
        loss_acm += loss.item()
        acc_acm += acc

        loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
        optimizer.step()

        if batch_acm%save_every == 0:
            torch.save({'model':residual_model.state_dict(), 'optimizer':optimizer.state_dict()},\
                    '%s/epoch%d_batch_%dacc_%.3f'%(model_path, train_data.epoch, batch_acm, acc_acm/print_every))

        if batch_acm%print_every == 0:
            print('epoch %d, batch_acm %d, loss %.3f, acc %.3f'\
                %(train_data.epoch, batch_acm, loss_acm/print_every, acc_acm/print_every), flush=True)
            print('lr: ', optimizer._rate)
            loss_acm, acc_acm = 0., 0.

