import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac

class ConvDownsampling_downsampleH(nn.Module):
    def __init__(self, channel_in):
        super(ConvDownsampling_downsampleH, self).__init__()

        self.channel_in = channel_in

        self.haar_weights = torch.ones(2, 1, 2, 1)
        self.haar_weights[0, 0, 0, 0] = 1/2
        self.haar_weights[0, 0, 1, 0] = 1/2

        self.haar_weights[1, 0, 0, 0] = -1/2
        self.haar_weights[1, 0, 1, 0] = 1/2

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = True

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=(2, 1), groups=self.channel_in)
            out = out.reshape([x.shape[0], self.channel_in, 2, x.shape[2] // 2, x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 2, x.shape[2] // 2, x.shape[3]])
            return out
        else:
            r = 2
            in_batch, in_channel, in_height, in_width = x.size()
            out_batch, out_channel, out_height, out_width = in_batch, int(
                in_channel / r), r * in_height, in_width

            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 2, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 2, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=(2, 1), groups=self.channel_in) * 2.0


class ConvDownsampling(nn.Module):
    def __init__(self, scale_W, scale_H):
        super(ConvDownsampling, self).__init__()
        self.scale_W = scale_W
        self.scale_H = scale_H
        self.scale2 = self.scale_H * self.scale_W

        self.conv_weights = torch.eye(self.scale2)

        if self.scale_W == self.scale_H == 2: # haar init
            self.conv_weights[0] = torch.Tensor([1./4, 1./4, 1./4, 1./4])
            self.conv_weights[1] = torch.Tensor([1./4, -1./4, 1./4, -1./4])
            self.conv_weights[2] = torch.Tensor([1./4, 1./4, -1./4, -1./4])
            self.conv_weights[3] = torch.Tensor([1./4, -1./4, -1./4, 1./4])
        else:
            self.conv_weights[0] = torch.Tensor([1./(self.scale2)] * (self.scale2))

        self.conv_weights = nn.Parameter(self.conv_weights)

    def forward(self, x, rev=False):
        if not rev:
            h = x.shape[2]
            w = x.shape[3]

            [B, C, H, W] = list(x.size())
            x = x.reshape(B, C, H // self.scale_H, self.scale_H, W // self.scale_W, self.scale_W)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(B, C * self.scale2, H // self.scale_H, W // self.scale_W)

            conv_weights = self.conv_weights.reshape(self.scale2, self.scale2, 1, 1)
            conv_weights = conv_weights.repeat(C, 1, 1, 1)

            out = F.conv2d(x, conv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale2, H // self.scale_H, W // self.scale_W)
            out = torch.transpose(out, 1, 2)
            out = out.reshape(B, C * self.scale2, H // self.scale_H, W // self.scale_W)


            return out
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.scale2, self.scale2, 1, 1)

            [B, C_, H_, W_] = list(x.size())
            C = C_ // self.scale2
            H = H_ * self.scale_H
            W = W_ * self.scale_W

            inv_weights = inv_weights.repeat(C, 1, 1, 1)

            x = x.reshape(B, self.scale2, C, H_, W_)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(B, C_, H_, W_)

            out = F.conv2d(x, inv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale_H, self.scale_W, H_, W_)
            out = out.permute(0, 1, 4, 2, 5, 3)
            out = out.reshape(B, C, H, W)

            return out

class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], down_num=2, use_ConvDownsampling=False, down_scale_W=2, down_scale_H=2):
        super(InvRescaleNet, self).__init__()

        operations = []

        if use_ConvDownsampling:
            down_num = 1

        current_channel = channel_in
        for i in range(down_num):
            if use_ConvDownsampling:
                b = ConvDownsampling(down_scale_W, down_scale_H)
                current_channel *= down_scale_W * down_scale_H
            elif down_scale_W==2 and down_scale_H==4 and i==1:
                b = ConvDownsampling_downsampleH(current_channel)
                current_channel *= 2
            else:
                b = HaarDownsampling(current_channel)
                current_channel *= 4
            operations.append(b)
            for j in range(block_num[i]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

