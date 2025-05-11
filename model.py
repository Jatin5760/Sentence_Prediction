import torch
from torch import nn


class ResidualModel(nn.Module):
    # 二层残差感知机
    def __init__(self, device, inp_size=4*768, output_size=768, \
        relu_layer_size=1024, num_residual_layers=2, dropout=0.5):
        super(ResidualModel, self).__init__()

        self.length_num = int(inp_size/768)
        self.layernorm1 = nn.LayerNorm(inp_size)
        self.fc1 = nn.Linear(inp_size, relu_layer_size)

        self.block = nn.Sequential(
            nn.Linear(relu_layer_size, relu_layer_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(relu_layer_size, relu_layer_size),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList()
        for i in range(num_residual_layers):
            self.layers.append(self.block)

        self.fc2 = nn.Linear(relu_layer_size, output_size)
        self.layernorm2 = nn.LayerNorm(output_size)
        self.device = device

    def CSloss(self, x, pred_y, y, bsz):
        # pred_y [bsz, 768], y [bsz, 768]
        # 向量长度归一化
        pred_y = pred_y / torch.norm(pred_y, dim=1).view(-1, 1)
        y = y / torch.norm(y, dim=1).view(-1, 1)
        # 正样本损失
        loss = -torch.sum(pred_y*y)

        # 所有负样本损失
        for i in range(bsz):
            pred_y_t = pred_y[i]
            y_t = y.clone()
            loss_t = torch.sum(pred_y_t * y_t, dim=1)
            loss_t = torch.sum(torch.exp(loss_t))
            loss += torch.log(loss_t)
        return loss

    def acc_compute(self, pred_y, y, bsz):
        acc = 0
        for i in range(bsz):
            idx = torch.sum(pred_y[i]*y, dim=1).argmax().item()
            if idx == i:
                acc += 1
        return acc/bsz

    def work(self, x):
        x = self.layernorm1(x)
        x = self.fc1(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.fc2(x)
        pred_y = self.layernorm2(x)
        return pred_y

    def forward(self, inp, y):
        bsz, inp_size = inp.size()
        x = self.layernorm1(inp)
        x = self.fc1(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.fc2(x)
        pred_y = self.layernorm2(x)
        loss = self.CSloss(inp, pred_y, y, bsz)
        acc = self.acc_compute(pred_y, y, bsz)
        return pred_y, loss, acc

