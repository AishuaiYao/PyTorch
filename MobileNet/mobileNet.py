'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2017, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${aslRecognition}

@File    :   ${mobileNet}.py

@Time    :   ${v1:2020-02-27} ${v2:2020-02-28}

@Desc    :   practice

'''

import torch.nn  as nn

class Block(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        nn.Module.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=inchannel,kernel_size = 3,stride = stride,padding=1,groups = inchannel),
            nn.BatchNorm2d(inchannel),
            nn.ReLU6()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel ,kernel_size=1,stride = 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )


    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'MobileNetV1'
        self.cfg = [64,(2,128),128,(2,256),256,(2,512),512,512,512,512,512,(2,1024),(2,1024)]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layers = self.make_layer(32)
        self.pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024,5)


    def make_layer(self,inchannel):
        layers = []
        for param in self.cfg:
            stride = 1 if isinstance(param,int) else param[0]
            outchannel = param if isinstance(param,int) else param[1]
            layers.append(Block(inchannel,outchannel,stride))
            inchannel = outchannel
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.pooling(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out



class InvertResidual(nn.Module):
    def __init__(self, inchannel, outchannel, expansion_scale, stride):
        nn.Module.__init__(self)
        self.expand_channel = expansion_scale * inchannel
        self.stride = stride
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=self.expand_channel,kernel_size=1,stride=1),
            nn.BatchNorm2d(self.expand_channel),
            nn.ReLU6()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.expand_channel,out_channels=self.expand_channel,kernel_size=3,
                      stride=stride, padding=1, groups=self.expand_channel),
            nn.BatchNorm2d(self.expand_channel),
            nn.ReLU6()
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=self.expand_channel,out_channels=outchannel,kernel_size=1,stride = 1),
            nn.BatchNorm2d(outchannel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self,x):
        out = self.expansion(x)
        out = self.depthwise(out)
        out = self.projection(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'MobileNetV2'
        self.cfgs = [
                    #t, c, n, s     [expansion_scale,out_channel,repeated times,stride]
                    [1, 16, 1, 1],
                    [6, 24, 2, 2],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1]]
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )
        self.bottlenecks = self.make_layer(32)
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels=320,out_channels=1280,kernel_size=1,stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(),
            nn.AvgPool2d(kernel_size=7),
            nn.Conv2d(in_channels=1280,out_channels=5,kernel_size=1)
        )


    def make_layer(self,inchannel):
        bottlenecks = []
        for t, c, n, s in self.cfgs:
            for i in range(n):
                bottlenecks.append(InvertResidual(inchannel=inchannel,outchannel=c,expansion_scale=t,stride=s))
                inchannel = c
        return nn.Sequential(*bottlenecks)


    def forward(self,x):
        out = self.head(x)
        out = self.bottlenecks(out)
        out = self.tail(out)
        out = out.squeeze(2)
        out = out.squeeze(2)
        return out








