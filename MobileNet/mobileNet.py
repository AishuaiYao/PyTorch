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
        self.cfg = [64,(2,128),128,(2,256),256,(2,512),512,512,512,512,512,(2,1024),(2,1024)]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layers = self.makelayer(32)
        self.pooling = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024,5)

        self.model_name = 'MobileNetV1'


    def makelayer(self,inchannel):
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














