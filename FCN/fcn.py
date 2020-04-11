'''
@Author  :   {AishuaiYao}
@License :   (C) Copyright 2013-2020, {None}
@Contact :   {aishuaiyao@163.com}
@Software:   ${segmentation}
@File    :   ${fcn}.py
@Time    :   ${2020-04-06}
@Desc    :   deconvlution experiment
'''

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
#
pretrained_net = models.vgg16(pretrained=True)
# a = pretrained_net.children()
# n = list(pretrained_net.children())[0]
# # d = len(n._modules)
# # b = n._modules['0']
# # c = nn.Conv2d(3,64,3,1,0,bias=False)
# e = []
# for i in range(31):
#     e.append(n._modules[str(i)])
#
# f = nn.Sequential(*e)
#
# #
# #
# # l = nn.Sequential(*list(pretrained_net.children())[:-1])
# print('x')

class FCN(nn.Module):
    def __init__(self,num_classes):
        super(FCN, self).__init__()
        conv_sequential= list(pretrained_net.children())[0]
        modules_list = []
        for i in range(17):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage1 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(17,24):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage2 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(24,31):
            modules_list.append(conv_sequential._modules[str(i)])
        modules_list.append(nn.Conv2d(in_channels=512,out_channels=4096,kernel_size=1,stride=1,padding=0))
        modules_list.append(nn.Conv2d(in_channels=4096,out_channels=4096,kernel_size=1,stride=1,padding=0))
        self.stage3 = nn.Sequential(*modules_list)

        self.scores3 = nn.Conv2d(in_channels=4096,out_channels=num_classes,kernel_size=1)
        self.scores2 = nn.Conv2d(in_channels=512,out_channels=num_classes,kernel_size=1)
        self.scores1 = nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1)

        # N=(w-1)��s+k-2p
        self.upsample_8x = nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes,kernel_size=16,stride=8,padding=4,bias= False)
        self.upsample_8x.weight.data = self.bilinear_kernel(num_classes,num_classes,16)
        self.upsample_16x = nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_16x.weight.data = self.bilinear_kernel(num_classes,num_classes,4)
        self.upsample_32x = nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes,kernel_size=4,stride=2,padding=1,bias=False)
        self.upsample_32x.weight.data = self.bilinear_kernel(num_classes,num_classes,4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores3(s3)
        s3 = self.upsample_32x(s3)

        s2 = self.scores2(s2)
        s2 = s2 + s3
        s2 = self.upsample_16x(s2)

        s1 = self.scores1(s1)
        s = s1 + s2
        s = self.upsample_8x(s)

        return s

    def bilinear_kernel(self,in_channels, out_channels, kernel_size):
        '''
        return a bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight)

