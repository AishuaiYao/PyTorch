'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2020, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${aslRecognition}

@File    :   ${free}.py

@Time    :   ${2020-03-31}

@Desc    :   practice

'''

import torch
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
from torch.utils import data
from torchvision import transforms,datasets
from FreePractice import myNet
from tools import *

batch_size = 16
classes = ['A','B','C','D','E']
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
train_data = datasets.ImageFolder('../data/asl_dataset/train',transform)#数据加载器
train_loader = data.DataLoader(train_data,batch_size,shuffle=True)
valid_data = datasets.ImageFolder('../data/asl_dataset/valid',transform)
valid_loader = data.DataLoader(valid_data,batch_size,shuffle = True)


model = myNet.CNN()
model.to(device)
summary(model,(3,224,224))
optimizer = torch.optim.Adam(model.parameters())


def train(model, device, train_loader,optimizer, epoch):
    model.train()

    loss = 0

    for batch_idx,(data,label) in enumerate(train_loader):
        data,label = data.to(device),label.to(device)

        optimizer.zero_grad()
        out =  model(data)
        loss = F.cross_entropy(out,label)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('epoch: %d/%d \t deal: %d/%d \t loss: %.4f' % (epoch, epochs, batch_idx * batch_size,
                                                                 len(train_loader.dataset), loss.item()))
    return loss.item()

def test(model, device, valid_loader):
    model.eval()

    loss =0
    accuracy = 0

    with torch.no_grad():
        for data, label in valid_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss += F.cross_entropy(out, label, reduction='sum').item()
            pred = out.max(1, keepdim=True)[1]
            accuracy += pred.eq(label.view_as(pred)).sum().item()
        total = len(valid_loader.dataset)
        loss /= total
        accuracy /= total

        print('valid loss: %.4f \t accuracy: %.2f%% \n' % (loss, 100 * accuracy))

        return loss


train_loss = []
val_loss = []
for epoch in range(1):
    train_loss.append(train(model, device, train_loader, optimizer, epoch))
    val_loss.append(test(model, device, valid_loader))

































