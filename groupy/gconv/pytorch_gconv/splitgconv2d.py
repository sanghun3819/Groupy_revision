import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *

make_indices_functions = {(1, 4): make_c4_z2_indices,
                          (4, 4): make_c4_p4_indices,
                          (1, 8): make_d4_z2_indices,
                          (8, 8): make_d4_p4m_indices,
                          (1, 16): make_d8_z2_indices,
                          (16, 16) : make_d8_p8m_indices}


def trans_filter(w, inds):
    #print('inds', type(inds), inds.shape)
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64) #576, 3
    #print('inds_reshape', type(inds_reshape), inds_reshape.shape)
    #print('w.shape', w.shape)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()] # 64 64 576
    #w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 1].tolist()]
    #print(inds_reshape[:, 0].shape)
    #print(inds_reshape[:, 1].shape)
    #print(inds_reshape[:, 2].shape)
    #print(inds_reshape[:, 0])
    #print(inds_reshape[:, 1])
    #print(inds_reshape[:, 2])
    #print('w_indexed.shape', w_indexed.shape)

    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3]) # 64 64 8 8 3 3
    #print(type(w_indexed), w_indexed.shape)
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5) #64 8 64 8 3 3
    #print(type(w_transformed), w_transformed.shape)
    return w_transformed.contiguous()

def trans_filter_16(w, inds):
    #print('inds', type(inds), inds.shape)
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64) #576, 3
    #print('inds_reshape', type(inds_reshape), inds_reshape.shape)
    #print('w.shape', w.shape)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()] # 64 64 576
    #w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 1].tolist()]
    #print(inds_reshape[:, 0].shape)
    #print(inds_reshape[:, 1].shape)
    #print(inds_reshape[:, 2].shape)
    #print(inds_reshape[:, 0])
    #print(inds_reshape[:, 1])
    #print(inds_reshape[:, 2])
    #print('w_indexed.shape', w_indexed.shape)

    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3]) # 64 64 8 8 3 3
    #print(type(w_indexed), w_indexed.shape)
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5) #64 8 64 8 3 3
    #print(type(w_transformed), w_transformed.shape)
    return w_transformed.contiguous()

class SplitGConv2D_SF(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D_SF, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        print('SplitGConv2d_SF init')
        self.weight1 = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        #print(torch.Tensor(
        #    out_channels, in_channels, self.input_stabilizer_size, *kernel_size).size())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds) #64 8 64 8 3 3
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape) #512 512 3 3

        tw_sf = trans_filter(self.weight1, self.inds) #64 8 64 8 3 3
        tw_sf = tw_sf.view(tw_shape)

        input_shape = input.size() # batch 64 8 9 9
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])
        #batch 512 w h
        y1 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding, dilation=1)
        batch_size, _, ny_out, nx_out = y1.size()
        y1 = y1.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y1.shape)

        tw_sf_0 = tw_sf.clone()
        # tw_sf_1 = tw.clone()
        # tw_sf_2 = tw.clone()
        # # tw_sf_3 = tw.clone()
        # tw_sf_4 = tw.clone()
        # # tw_sf_5 = tw.clone()
        # tw_sf_6 = tw.clone()

        a0 = tw_sf[..., 0, 0]
        a1 = tw_sf[..., 0, 1]
        a2 = tw_sf[..., 0, 2]
        a3 = tw_sf[..., 1, 0]
        a4 = tw_sf[..., 1, 1]
        a5 = tw_sf[..., 1, 2]
        a6 = tw_sf[..., 2, 0]
        a7 = tw_sf[..., 2, 1]
        a8 = tw_sf[..., 2, 2]

        tw_sf_0[..., 0, 0] = a3
        tw_sf_0[..., 1, 0] = a6
        tw_sf_0[..., 2, 0] = a7
        tw_sf_0[..., 0, 1] = a0
        tw_sf_0[..., 1, 1] = a4
        tw_sf_0[..., 2, 1] = a8
        tw_sf_0[..., 0, 2] = a1
        tw_sf_0[..., 1, 2] = a2
        tw_sf_0[..., 2, 2] = a5

        tw_sf = tw_sf_0

        # tw_sf_1[..., 0, 0] = a6
        # tw_sf_1[..., 1, 0] = a7
        # tw_sf_1[..., 2, 0] = a8
        # tw_sf_1[..., 0, 1] = a3
        # tw_sf_1[..., 1, 1] = a4
        # tw_sf_1[..., 2, 1] = a5
        # tw_sf_1[..., 0, 2] = a0
        # tw_sf_1[..., 1, 2] = a1
        # tw_sf_1[..., 2, 2] = a2

        # tw_sf_2[..., 0, 0] = a7
        # tw_sf_2[..., 1, 0] = a8
        # tw_sf_2[..., 2, 0] = a5
        # tw_sf_2[..., 0, 1] = a6
        # tw_sf_2[..., 1, 1] = a4
        # tw_sf_2[..., 2, 1] = a2
        # tw_sf_2[..., 0, 2] = a3
        # tw_sf_2[..., 1, 2] = a0
        # tw_sf_2[..., 2, 2] = a1


        # tw_sf_3[..., 0, 0] = a8
        # tw_sf_3[..., 1, 0] = a5
        # tw_sf_3[..., 2, 0] = a2
        # tw_sf_3[..., 0, 1] = a7
        # tw_sf_3[..., 1, 1] = a4
        # tw_sf_3[..., 2, 1] = a1
        # tw_sf_3[..., 0, 2] = a6
        # tw_sf_3[..., 1, 2] = a3
        # tw_sf_3[..., 2, 2] = a0

        # tw_sf_4[..., 0, 0] = a5
        # tw_sf_4[..., 1, 0] = a2
        # tw_sf_4[..., 2, 0] = a1
        # tw_sf_4[..., 0, 1] = a8
        # tw_sf_4[..., 1, 1] = a4
        # tw_sf_4[..., 2, 1] = a0
        # tw_sf_4[..., 0, 2] = a7
        # tw_sf_4[..., 1, 2] = a6
        # tw_sf_4[..., 2, 2] = a3

        # tw_sf_5[..., 0, 0] = a2
        # tw_sf_5[..., 1, 0] = a1
        # tw_sf_5[..., 2, 0] = a0
        # tw_sf_5[..., 0, 1] = a5
        # tw_sf_5[..., 1, 1] = a4
        # tw_sf_5[..., 2, 1] = a3
        # tw_sf_5[..., 0, 2] = a8
        # tw_sf_5[..., 1, 2] = a7
        # tw_sf_5[..., 2, 2] = a6

        # tw_sf_6[..., 0, 0] = a1
        # tw_sf_6[..., 1, 0] = a0
        # tw_sf_6[..., 2, 0] = a3
        # tw_sf_6[..., 0, 1] = a2
        # tw_sf_6[..., 1, 1] = a4
        # tw_sf_6[..., 2, 1] = a6
        # tw_sf_6[..., 0, 2] = a5
        # tw_sf_6[..., 1, 2] = a8
        # tw_sf_6[..., 2, 2] = a7

        y2 = F.conv2d(input, weight=tw_sf, bias=None, stride=self.stride,
                        padding=self.padding, dilation=1)
        batch_size, _, ny_out, nx_out = y2.size()
        y2 = y2.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y2.shape)

        # y3 = F.conv2d(input, weight=tw_sf_1, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y3.size()
        # y3 = y3.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y4 = F.conv2d(input, weight=tw_sf_2, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y4.size()
        # y4 = y4.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y5 = F.conv2d(input, weight=tw_sf_3, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y5.size()
        # y5 = y5.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y6 = F.conv2d(input, weight=tw_sf_4, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y6.size()
        # y6 = y6.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y7 = F.conv2d(input, weight=tw_sf_5, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y7.size()
        # y7 = y7.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y8 = F.conv2d(input, weight=tw_sf_6, bias=None, stride=self.stride,
        #               padding=self.padding, dilation=1)
        # batch_size, _, ny_out, nx_out = y8.size()
        # y8 = y8.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        # y_list = [y1, y2, y4, y6, y8]
        # y_list = [y1, y2, y3, y4, y5, y6, y7, y8]
        y_list = [y1, y2]
        #print('debug0', y1.shape, y2.shape, y3.shape)
        #y = torch.cat((y1, y2, y3), dim=0)
        y = torch.mean(torch.stack(y_list), dim=0)
        #print('debug1', y.shape)


        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y

class SplitGConv2D_SC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D_SC, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        #print(torch.Tensor(
        #    out_channels, in_channels, self.input_stabilizer_size, *kernel_size).size())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds) #64 8 64 8 3 3
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape) #512 512 3 3
        input_shape = input.size() # batch 64 8 9 9
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])
        #batch 512 w h
        y1 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding, dilation=1)
        batch_size, _, ny_out, nx_out = y1.size()
        y1 = y1.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y1.shape)

        y2 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=2, dilation=2)
        batch_size, _, ny_out, nx_out = y2.size()
        y2 = y2.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y2.shape)

        y3 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                      padding=3, dilation=3)
        batch_size, _, ny_out, nx_out = y3.size()
        y3 = y3.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y3.shape)

        y_list = [y1, y2, y3]
        #print('debug0', y1.shape, y2.shape, y3.shape)
        #y = torch.cat((y1, y2, y3), dim=0)
        y = torch.mean(torch.stack(y_list), dim=0)
        #print('debug1', y.shape)


        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y

class SplitGConv2D_SCC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D_SCC, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        #print(torch.Tensor(
        self.weight1 = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        self.weight2 = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        #    out_channels, in_channels, self.input_stabilizer_size, *kernel_size).size())
        self.conv_1x1 = nn.Conv2d(in_channels= out_channels*output_stabilizer_size*3, out_channels= out_channels*output_stabilizer_size, kernel_size= 1, bias=None, stride=1, padding=0)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds) #64 8 64 8 3 3
        tw_scc_01 = trans_filter(self.weight1, self.inds) #64 8 64 8 3 3
        tw_scc_02 = trans_filter(self.weight2, self.inds)  # 64 8 64 8 3 3

        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)


        tw = tw.view(tw_shape) #512 512 3 3
        tw_scc_1 = tw_scc_01.view(tw_shape)
        tw_scc_2 = tw_scc_02.view(tw_shape)

        #tw_scc_01 = tw.clone()
        #tw_scc_02 = tw.clone()


        input_shape = input.size() # batch 64 8 9 9
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])
        #batch 512 w h
        y1 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding, dilation=1)
        batch_size, _, ny_out, nx_out = y1.size()
        #tmp = y1.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(tmp.shape)

        y2 = F.conv2d(input, weight=tw_scc_1, bias=None, stride=self.stride,
                        padding=2, dilation=2)
        #batch_size, _, ny_out, nx_out = y2.size()
        #y2 = y2.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y2.shape)

        y3 = F.conv2d(input, weight=tw_scc_2, bias=None, stride=self.stride,
                      padding=3, dilation=3)
        #batch_size, _, ny_out, nx_out = y3.size()
        #y3 = y3.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print(y3.shape)

        y_list = [y1, y2, y3]
        #print('debug0', y1.shape, y2.shape, y3.shape)
        y = torch.cat((y1, y2, y3), dim=1)
        #print('debug1', y.shape)
        y_shape = y.size()
        y = y.view(y_shape[0], -1, y_shape[-2], y_shape[-1])
        #print('debug1', y.shape)
        y = self.conv_1x1(y)
        #print('debug2', y.shape)
        #y = torch.mean(torch.stack(y_list), dim=0)
        y = y.view(-1, self.out_channels, self.output_stabilizer_size, y_shape[-2], y_shape[-1])
        #print('debug2', y.shape)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y

class SplitGConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4):
        super(SplitGConv2D, self).__init__()
        assert (input_stabilizer_size, output_stabilizer_size) in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        #print(torch.Tensor(
        #    out_channels, in_channels, self.input_stabilizer_size, *kernel_size).size())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[(self.input_stabilizer_size, self.output_stabilizer_size)](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds) #64 8 64 8 3 3
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape) #512 512 3 3
        input_shape = input.size() # batch 64 8 9 9
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])
        #batch 512 w h
        y1 = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding, dilation=1)
        #print('y1', y1.shape)
        batch_size, _, ny_out, nx_out = y1.size()
        y1 = y1.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)
        #print('y1', y1.shape)


        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y1 = y1 + bias

        return y1

class P4ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, *args, **kwargs)


class P4ConvP4(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, *args, **kwargs)


class P4MConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4MConvP4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M_SC(SplitGConv2D_SC):

    def __init__(self, *args, **kwargs):
        super(P4MConvP4M_SC, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)

class P4MConvP4M_SF(SplitGConv2D_SF):

    def __init__(self, *args, **kwargs):
        super(P4MConvP4M_SF, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)


class P4MConvP4M_SCC(SplitGConv2D_SCC):

    def __init__(self, *args, **kwargs):
        super(P4MConvP4M_SCC, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)


class P4MConvPE4M(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4MConvPE4M, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, *args, **kwargs)


#class P8MConvZ2(SplitGConv2D_test):

#    def __init__(self, *args, **kwargs):
#        super(P8MConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=16, *args, **kwargs)


#class P8MConvP8M(SplitGConv2D_test):

#    def __init__(self, *args, **kwargs):
#        super(P8MConvP8M, self).__init__(input_stabilizer_size=16, output_stabilizer_size=16, *args, **kwargs)