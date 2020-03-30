'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2017, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${aslRecognition}

@File    :   ${asl}.py

@Time    :   ${2020-02-27} ${10:14}

@Desc    :   practice

'''

import torch
import argparse
#在终端运行脚本时需要对模块的索引位置进行指定,第一个append是指向上级目录，第二个append是指回来
import sys
sys.path.append('..')
import tools
sys.path.append('./MobileNet')

import torch.nn.functional as F
from MobileNet.mobileNet import *
from torchsummary import summary
from torch.utils import data
from torchvision import datasets,transforms


parse = argparse.ArgumentParser('arguments list')
parse.add_argument('-b','--batch_size',default=32,type = int,help = 'batch size default is 16')
parse.add_argument('-e','--epochs',default = 10,type = int,help = 'epochs default is 10')
parse.add_argument('-sm','--model_save_path',default =  './model/*.pkl',type = str,
                   help = 'an example: "./model/*.pkl",the program will replace "*" with "model.model_name"')
parse.add_argument('-sf','--figure_save_path',default= './figure/loss_curve.png',type = str,help = 'the path of figure')
parse.add_argument('-v','--version',default='V3',type = str, help= 'switch net version,input V1 V2 or V3 please')
parse.add_argument('-t','--v3net_mode',default='large',type = str,help ='the v3net mode,input large or small please')
args = parse.parse_args()


transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
train_data = datasets.ImageFolder('../data/asl_dataset/train',transform = transform)
train_loader = data.DataLoader(train_data,batch_size = args.batch_size,shuffle=True)
test_data = datasets.ImageFolder('../data/asl_dataset/valid',transform = transform)
test_loader = data.DataLoader(test_data,batch_size=args.batch_size,shuffle=True)


version = args.version
model = MobileNetV1() if version=='V1' else MobileNetV2() if version=='V2' else MobileNetV3(args.v3net_mode)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model,(3,224,224))
optimizer = torch.optim.Adam(model.parameters())


def train(model,device,train_loader,optimizer,epoch):
    model.train()
    loss = 0
    for batch_idx,(data,label) in enumerate(train_loader):
        data,label = data.to(device),label.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out,label)
        loss.backward()
        optimizer.step()
        if batch_idx%20 == 0:
            print('epoch: %d/%d \t deal: %d/%d \t loss: %.4f'%(epoch,args.epochs,batch_idx*args.batch_size,
                                                               len(train_loader.dataset),loss.item()))
    return loss.item()

def test(model,device,test_loader):
    model.eval()
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for data,label in test_loader:
            data,label = data.to(device),label.to(device)
            out = model(data)
            loss += F.cross_entropy(out,label,reduction='sum').item()
            pred = out.max(1,keepdim = True)[1]
            accuracy += pred.eq(label.view_as(pred)).sum().item()
    total = len(test_loader.dataset)
    loss /= total
    accuracy /= total
    if accuracy>0.97:
        torch.save(model,args.model_save_path.replace('*',model.model_name))
    print('valid loss: %.4f \t accuracy: %.2f%% \n'%(loss,100*accuracy))

    return loss


train_loss = []
val_loss = []
for epoch in range(1,args.epochs+1):
    train_loss.append(train(model,device,train_loader,optimizer,epoch))
    val_loss.append(test(model,device,test_loader))

tools.show_loss(train_loss,val_loss,args.figure_save_path)