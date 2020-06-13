import os
import pdb
import sys
import torch
import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.models.pointlstm import PointLSTMCell


class TestPointLSTM(nn.Module):
    def __init__(self):
        super(TestPointLSTM, self).__init__()
        self.lstm = PointLSTMCell(pts_num=64, in_channels=132, hidden_dim=256,
                                  offset_dim=4, bias=True)

    def forward(self, inputs, hidden, cell_state):
        output = self.lstm(inputs, hidden, cell_state)
        return output[0], output[1]


module = TestPointLSTM().float().cuda()
module.eval()
features = torch.Tensor(1, 132, 64, 16).random_(10000).float().cuda() / 100
hidden_state = torch.Tensor(1, 260, 64, 16).random_(10000).float().cuda() / 100
cell_state = torch.Tensor(1, 256, 64, 16).random_(10000).float().cuda() / 100
hidden1, cell1 = module(features.clone(), hidden_state.clone(), cell_state.clone())

ind = torch.randperm(features.shape[2])

features = features[:, :, ind]
hidden_state = hidden_state[:, :, ind]
cell_state = cell_state[:, :, ind]
hidden2, cell2 = module(features, hidden_state, cell_state)

print((hidden1[:, :, ind] - hidden2)[0, :, :, 0].sum(0))
assert (hidden1[:, :, ind] - hidden2).max() < 1e-7, "f(transform(x))!=transform(f(x)"
print("f(transform(x))==transform(f(x)")
