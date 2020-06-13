import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, out_channel, dimension, with_bn=True):
        super(MLPBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                            nn.BatchNorm1d(out_channel[idx]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        )
                    )
        elif dimension == 2:
            for idx, channels in enumerate(out_channel[:-1]):
                if with_bn:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                            nn.BatchNorm2d(out_channel[idx + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.layer_list.append(
                        nn.Sequential(
                            nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        )
                    )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        for layer in self.layer_list:
            output = layer(output)
        return output


class MotionBlock(nn.Module):
    def __init__(self, out_channel, dimension, embedding_dim):
        super(MotionBlock, self).__init__()
        self.layer_list = []
        if dimension == 1:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channel[-1], kernel_size=1),
                    nn.BatchNorm1d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv1d(channels, out_channel[idx + 1], kernel_size=1),
                        nn.BatchNorm1d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        elif dimension == 2:
            self.layer_list.append(
                nn.Sequential(
                    nn.Conv2d(embedding_dim, out_channel[-1], kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channel[-1]),
                    nn.ReLU(inplace=True),
                )
            )
            for idx, channels in enumerate(out_channel[:-1]):
                self.layer_list.append(
                    nn.Sequential(
                        nn.Conv2d(channels, out_channel[idx + 1], kernel_size=(1, 1)),
                        nn.BatchNorm2d(out_channel[idx + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, output):
        position_embedding = self.layer_list[0](output[:, :4])
        feature_embedding = output[:, 4:]
        for layer in self.layer_list[1:]:
            feature_embedding = layer(feature_embedding)
        return position_embedding * feature_embedding


class GroupOperation(object):
    def __init__(self):
        pass

    def group_points(self, distance_dim, array1, array2, knn, dim):
        matrix, a1, a2 = self.array_distance(array1, array2, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        offsets = array1.unsqueeze(dim + 1) - neighbor
        offsets[:, :3] /= torch.sum(offsets[:, :3] ** 2, dim=1).unsqueeze(1) ** 0.5 + 1e-8
        return offsets

    def st_group_points(self, array, interval, distance_dim, knn, dim):
        batchsize, channels, timestep, num_pts = array.shape
        if interval // 2 > 0:
            array_padded = torch.cat((array[:, :, 0].unsqueeze(2).expand(-1, -1, interval // 2, -1),
                                      array,
                                      array[:, :, -1].unsqueeze(2).expand(-1, -1, interval // 2, -1)
                                      ), dim=2)
        else:
            array_padded = array
        neighbor_points = torch.zeros(batchsize, channels, timestep, num_pts * interval).to(array.device)
        for i in range(timestep):
            neighbor_points[:, :, i] = array_padded[:, :, i:i + interval].view(batchsize, channels, -1)
        matrix, a1, a2 = self.array_distance(array, neighbor_points, distance_dim, dim)
        dists, inputs_idx = torch.topk(matrix, knn, -1, largest=False, sorted=True)
        neighbor = a2.gather(-1, inputs_idx.unsqueeze(1).
                             expand(dists.shape[:1] + (a2.shape[1],) + dists.shape[1:]))
        array = array.unsqueeze(-1).expand_as(neighbor)
        ret_features = torch.cat((array[:, :4] - neighbor[:, :4], array[:, 4:], neighbor[:, 4:]), dim=1)
        # ret_features = torch.cat((array[:, :4] - neighbor[:, :4], neighbor[:, 4:]), dim=1)
        return ret_features

    def array_distance(self, array1, array2, dist, dim):
        # return array1.shape[-1] * array2.shape[-1] matrix
        distance_mat = array1.unsqueeze(dim + 1)[:, dist] - array2.unsqueeze(dim)[:, dist]
        mat_shape = distance_mat.shape
        mat_shape = mat_shape[:1] + (array1.shape[1],) + mat_shape[2:]
        array1 = array1.unsqueeze(dim + 1).expand(mat_shape)
        array2 = array2.unsqueeze(dim).expand(mat_shape)
        distance_mat = torch.sqrt((distance_mat ** 2).sum(1))
        return distance_mat, array1, array2
