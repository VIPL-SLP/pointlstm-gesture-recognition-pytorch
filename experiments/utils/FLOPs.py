import pdb
import torch


class CalculateParasAndFLOPs(object):
    def __init__(self):
        self.paras = 0
        self.FLOPs = 0

    def reset(self):
        self.paras = 0
        self.FLOPs = 0

    def calculate_all(self, model, inputs):
        self.Calculate(model)
        model(inputs)
        self.FLOPs /= 2
        self.show_params_FLOPs()

    def show_params_FLOPs(self):
        if self.paras < 1e5:
            print("Parameters: {}".format(self.paras))
        elif self.paras < 1e8:
            print("Parameters: {:.1f}M".format(self.paras / 1e6))
        else:
            print("Parameters: {:.1f}B".format(self.paras / 1e9))

        if self.FLOPs < 1e5:
            print("FLOPs: {}".format(self.FLOPs))
        else:
            print("FLOPs: {:.1f}M".format(self.FLOPs / 1e6))

    def conv_hook3(self, *inputs):
        _, channels, time, height, width = inputs[1][0].shape
        _, output_channels, output_time, output_height, output_width = inputs[2].shape
        t, h, w = inputs[0].kernel_size
        kernel_ops = channels * t * h * w
        bias_ops = 1 if inputs[0].bias is not None else 0
        self.paras += output_channels * (kernel_ops + bias_ops)
        self.FLOPs += output_channels * (kernel_ops * 2 + bias_ops) * output_time * output_height * output_width

    def conv_hook2(self, *inputs):
        _, channels, height, width = inputs[1][0].shape
        _, output_channels, output_height, output_width = inputs[2].shape
        if isinstance(inputs[0].kernel_size, int):
            h = inputs[0].kernel_size
            w = inputs[0].kernel_size
        else:
            h, w = inputs[0].kernel_size
        kernel_ops = channels * h * w
        bias_ops = 1 if inputs[0].bias is not None else 0
        self.paras += output_channels * (kernel_ops + bias_ops)
        self.FLOPs += output_channels * (kernel_ops * 2 + bias_ops) * output_height * output_width

    def linear_hook(self, *inputs):
        weight_ops = inputs[0].weight.nelement()
        bias_ops = inputs[0].bias.nelement()
        self.paras += weight_ops + bias_ops
        self.FLOPs += weight_ops * 2 + bias_ops

    def bn_hook(self, *inputs):
        self.paras += inputs[0].num_features * 2
        self.FLOPs += inputs[0].num_features * 2

    def relu_hook(self, *inputs):
        self.FLOPs += inputs[1][0].nelement()

    def pooling_hook2(self, *inputs):
        _, channels, height, width = inputs[1][0].shape
        _, output_channels, output_height, output_width = inputs[2].shape
        if isinstance(inputs[0].kernel_size, int):
            h = inputs[0].kernel_size
            w = inputs[0].kernel_size
        else:
            h, w = inputs[0].kernel_size
        kernel_ops = h * w
        bias_ops = 0
        self.FLOPs += output_channels * (kernel_ops + bias_ops) * output_height * output_width

    def pooling_hook3(self, *inputs):
        _, channels, time, height, width = inputs[1][0].shape
        _, output_channels, output_time, output_height, output_width = inputs[2].shape
        t, h, w = inputs[0].kernel_size
        kernel_ops = t * h * w
        bias_ops = 0
        self.FLOPs += output_channels * (kernel_ops + bias_ops) * output_time * output_height * output_width

    def Calculate(self, net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(self.conv_hook3)
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(self.conv_hook2)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(self.linear_hook)
            if isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(self.bn_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(self.bn_hook)
            # if isinstance(net, torch.nn.BatchNorm1d):
            #     net.register_forward_hook(self.bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(self.relu_hook)
            if isinstance(net, torch.nn.MaxPool3d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(self.pooling_hook3)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(self.pooling_hook2)
        for c in childrens:
            self.Calculate(c)
