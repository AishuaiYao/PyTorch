'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2017, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${mnistRecognition}

@File    :   ${resnet}.py

@Time    :   ${2020-02-22}

@Desc    :   practice

'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,3)
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out,2,2)

        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size,-1)#torch.view: 可以改变张量的维度和大小,与Numpy的reshape类似

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim = 1)
        return out


class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel,s = 1):
        nn.Module.__init__(self)
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride = s,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride = 1,padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if s != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride =s),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self,x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,residualBlock=BasicBlock,n_class=10):
        nn.Module.__init__(self)
        self.inchannel = 64
        self.conv1  = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride = 2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )
        self.pooling = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.layer1 = self.maker_layer(residualBlock,64,2,s = 1)
        self.layer2 = self.maker_layer(residualBlock,128,2,s = 2)
        self.layer3 = self.maker_layer(residualBlock,256,2,s = 2)
        self.layer4 = self.maker_layer(residualBlock,512,2,s = 2)
        self.fc = nn.Linear(512,n_class)


    def maker_layer(self,block,channels,n_blocks,s):
        strides = [s]+[1]*(n_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel,channels,stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    def forward(self,x):
        out = self.conv1(x)
        out = self.pooling(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.fc(out)

        return out



