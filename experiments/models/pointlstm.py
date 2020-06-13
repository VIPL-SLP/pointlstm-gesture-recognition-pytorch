import pdb
import torch
import numpy as np
import torch.nn as nn


class PointLSTMCell(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, bias):
        super(PointLSTMCell, self).__init__()
        self.bias = bias
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.offset_dim = offset_dim
        self.pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((None, 1))
        )
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.offset_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=(1, 1),
                              bias=self.bias)

    def forward(self, input_tensor, hidden_state, cell_state):
        hidden_state[:, :4] -= input_tensor[:, :4]
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * torch.tanh(c_next)
        return self.pool(h_next), self.pool(c_next)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1).cuda())


class PointLSTM(nn.Module):
    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, num_layers, topk=16, offsets=False,
                 batch_first=True, bias=True, return_all_layers=False):
        super(PointLSTM, self).__init__()
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.bias = bias
        self.topk = topk
        self.offsets = offsets
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.offset_dim = offset_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_dim[i - 1] + 4
            cell_list.append(PointLSTMCell(pts_num=self.pts_num,
                                           in_channels=cur_in_channels,
                                           hidden_dim=self.hidden_dim[i],
                                           offset_dim=self.offset_dim,
                                           bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        # batch, timestep, c, n (N points, M neighbor)
        if not self.batch_first:
            # (t, b, c, n) -> (b, t, c, n)
            input_tensor = input_tensor.permute(1, 0, 2, 3)
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []
        position = input_tensor[:, :, :4]
        if self.offsets:
            centroids = torch.mean(position[:, :, :3], dim=3)
            group_offsets = (centroids[:, :-1] - centroids[:, 1:])[:, :, :, None]
            group_ind = torch.cat(
                (
                    self.group_points(position[:, 0, :3], position[:, 0, :3], dim=2,
                                      topk=self.topk).unsqueeze(1),
                    self.group_points(position[:, 1:, :3] + group_offsets, position[:, :-1, :3],
                                      dim=3, topk=self.topk),
                ),
                dim=1
            )
        else:
            group_ind = torch.cat(
                (
                    self.group_points(position[:, 0, :3], position[:, 0, :3], dim=2,
                                      topk=self.topk).unsqueeze(1),
                    self.group_points(position[:, 1:, :3], position[:, :-1, :3],
                                      dim=3, topk=self.topk),
                ),
                dim=1
            )
        seq_len = input_tensor.shape[1]

        cur_layer_input = input_tensor.unsqueeze(-1)
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                past = 0 if t == 0 else t - 1
                center_pts = cur_layer_input[:, t].expand(-1, -1, -1, self.topk)
                h_with_pos = torch.cat((position[:, past].unsqueeze(-1), h), dim=1)
                h_grouped = h_with_pos.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).
                           expand(-1, -1, self.hidden_dim[layer_idx] + self.offset_dim, -1)) \
                    .permute(0, 2, 1, 3)
                c_grouped = c.squeeze(-1).unsqueeze(1).expand(-1, self.pts_num, -1, -1). \
                    gather(3, group_ind[:, t].unsqueeze(2).expand(-1, -1, self.hidden_dim[layer_idx], -1)) \
                    .permute(0, 2, 1, 3)
                h, c = self.cell_list[layer_idx](
                    input_tensor=center_pts.clone(),
                    hidden_state=h_grouped.clone(),
                    cell_state=c_grouped.clone()
                )
                output_inner.append(h)
            layer_output = torch.cat((position.unsqueeze(-1), torch.stack(output_inner, dim=1)), dim=2)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list, group_ind

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def group_points(self, array1, array2, dim, topk):
        dist, _, _ = self.array_distance(array1, array2, dim)
        dists, idx = torch.topk(dist, topk, -1, largest=False, sorted=False)
        return idx

    @staticmethod
    def array_distance(array1, array2, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1) - array2.unsqueeze(dim)
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(dim - 1))
        return distance_mat, array1, array2

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    @staticmethod
    def tensor2numpy(tensor, name="test"):
        np.save(name, tensor.cpu().detach().numpy())
