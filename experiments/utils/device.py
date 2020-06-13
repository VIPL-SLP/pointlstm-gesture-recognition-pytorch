import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        device = str(device)
        if device != 'None':
            self.gpu_list = [i for i in range(len(device.split(',')))]
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            output_device = self.gpu_list[0]
            self.occupy_gpu(self.gpu_list)
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def load_weights(self, model, weights_path, ignore_weights):
        print('Load weights from {}.'.format(weights_path))
        try:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        except RuntimeError:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))['model_state_dict']
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    @staticmethod
    def occupy_gpu(gpus=None):
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
