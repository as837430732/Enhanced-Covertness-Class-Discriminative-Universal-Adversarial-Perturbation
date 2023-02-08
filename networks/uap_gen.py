from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def uap(in_planes):
    # The universal adversarial representation is represented as a 1x1 convolution
    return nn.Conv2d(in_planes, in_planes,
                     kernel_size=1, stride=1,
                     padding=0, bias=False)


class UAP_gen(nn.Module):
    def __init__(self,
                 epsilon,
                 shape=(224, 224),
                 num_channels=3,
                 mean=[0., 0., 0.],
                 std=[1., 1., 1.],
                 batch_size=4,
                 ngf=64,
                 use_cuda=True,
                 gpu_ids=[]):
        super(UAP_gen, self).__init__()

        self.use_cuda = use_cuda
        self.epsilon = epsilon

        im_noise = np.random.uniform(0, 1, size=(batch_size, num_channels, *shape))
        noise_t = torch.from_numpy(im_noise).type(torch.FloatTensor)
        if use_cuda:
            noise_t = noise_t.cuda()
        self.z = nn.Parameter(noise_t, requires_grad=True)


        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:, idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:, idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()

        self.RG = ResnetGenerator(num_channels, num_channels, ngf, norm_type='batch', act_type='relu', gpu_ids=gpu_ids)

    def forward(self, x, ):

        uap = self.RG(self.z)
        t_uap = uap[:x.size(0), :, :, :].cuda()
        #t_uap = uap[0, :, :, :].cuda()
        # Project to l-infinity ball
        t_uap.data = torch.clamp(t_uap.data, -self.epsilon, self.epsilon)
        # Put image into original form
        orig_img = x * self.std_tensor + self.mean_tensor
        # Add uap to input
        adv_orig_img = orig_img + t_uap
        # Put image into normalized form
        adv_x = (adv_orig_img - self.mean_tensor) / self.std_tensor

        return adv_x


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type, act_type='selu', use_dropout=False, n_blocks=6,
                 padding_type='reflect', gpu_ids=[]):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        self.name = 'resnet'
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpulist = gpu_ids
        self.num_gpus = len(self.gpulist)

        use_bias = norm_type == 'instance'

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                            bias=use_bias),
                  norm_layer(ngf),
                  self.act]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model0 += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * mult * 2),
                       self.act]

        if self.num_gpus == 1:
            mult = 2 ** n_downsampling
            for i in range(n_blocks):
                model0 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
        elif self.num_gpus == 2:
            model1 = []
            mult = 2 ** n_downsampling
            mid = int(n_blocks / 2)
            for i in range(mid):
                model0 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
            for i in range(n_blocks - mid):
                model1 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
        elif self.num_gpus == 3:
            model1 = []
            model2 = []
            mult = 2 ** n_downsampling
            mid1 = int(n_blocks / 5)
            mid2 = mid1 + int((n_blocks - mid1) / 4.0 * 3)
            # mid = int(n_blocks / 2)
            for i in range(mid1):
                model0 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
            for i in range(mid1, mid2):
                model2 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]
            for i in range(mid2, n_blocks):
                model1 += [
                    ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                use_bias=use_bias)]

        if self.num_gpus >= 2:
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model1 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias),
                           norm_layer(int(ngf * mult / 2)),
                           self.act]
            model1 += [nn.ReflectionPad2d(3)]
            model1 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model1 += [nn.Tanh()]
        else:
            for i in range(n_downsampling):
                mult = 2 ** (n_downsampling - i)
                model0 += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias),
                           norm_layer(int(ngf * mult / 2)),
                           self.act]
            model0 += [nn.ReflectionPad2d(3)]
            model0 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
            model0 += [nn.Tanh()]

        self.model0 = nn.Sequential(*model0)
        self.model0.cuda(self.gpulist[0])
        if self.num_gpus == 2:
            self.model1 = nn.Sequential(*model1)
            self.model1.cuda(self.gpulist[1])
        if self.num_gpus == 3:
            self.model2 = nn.Sequential(*model2)
            self.model2.cuda(self.gpulist[2])

    def forward(self, input):
        input = input.cuda(self.gpulist[0])
        input = self.model0(input)
        if self.num_gpus == 3:
            input = input.cuda(self.gpulist[2])
            input = self.model2(input)
        if self.num_gpus == 2:
            input = input.cuda(self.gpulist[1])
            input = self.model1(input)

        return input

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
