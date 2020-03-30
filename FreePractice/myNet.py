'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2020, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${aslRecognition}

@File    :   ${myNet}.py

@Time    :   ${2020-03-31}

@Desc    :   practice

'''

import torch.nn as nn

args = [[3, 32, 2,1],
        [32, 64, 2, 1],
        [64, 128, 2, 1],
        [128, 256, 1, 0]]


def build_net(args):
    modules = nn.ModuleList()

    for i, arg in enumerate(args):
        module = nn.Sequential()
        conv = nn.Conv2d(in_channels=arg[0],out_channels=arg[1],kernel_size=3,stride=arg[2],padding=arg[3])
        module.add_module('conv_%d'%i,conv)
        bn = nn.BatchNorm2d(arg[1])
        module.add_module('bn_%d'%i,bn)
        act = nn.LeakyReLU(0.1,inplace=True)
        module.add_module('act_%d'%i,act)

        modules.append(module)

    module = nn.Sequential()
    linear = nn.Linear(in_features=173056, out_features=512)
    module.add_module('fc1', linear)
    act = nn.LeakyReLU(0.1,inplace=True)
    module.add_module('act5', act)

    linear = nn.Linear(in_features=512, out_features=5)
    module.add_module('fc2', linear)
    modules.append(module)


    return modules



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.module_list = build_net(args)

    def forward(self, x):
        for i in range(5):
            if i == 4:
                x = x.view(x.size(0),-1)
            x = self.module_list[i](x)
        return x












