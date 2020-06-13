import os
import pdb
import sys
import torch
import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.models.pointlstm import PointLSTM


class TestPointLSTM(nn.Module):
    def __init__(self, topk=16):
        super(TestPointLSTM, self).__init__()
        self.lstm = PointLSTM(pts_num=64, in_channels=132, hidden_dim=256,
                              offset_dim=4, num_layers=1, topk=topk)

    def forward(self, inputs, offsets):
        output = self.lstm(inputs, offsets)
        return output[0][0].squeeze(-1), output[2]


module = TestPointLSTM().float().cuda()
ind = torch.randperm(64)

module.eval()
inputs = torch.Tensor(1, 32, 132, 64).random_(10000).float().cuda() / 100
offsets = torch.Tensor(1, 32, 3, 64).random_(10000).float().cuda() / 100
output1, group_id1 = module(inputs, offsets)

inputs = inputs[:, :, :, ind]
offsets = offsets[:, :, :, ind]
output2, group_id2 = module(inputs, offsets)

for i in range(32):
    print("Timestep:", i, (output1[0, i, :, ind] - output2[0, i]).max())

assert (output1[:, :, :, ind] - output2).max() < 1e-7, "f(transform(x))!=transform(f(x))"
print("f(transform(x))==transform(f(x))")
