'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2017, {None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${aslRecognition}

@File    :   ${demo}.py

@Time    :   ${2020-02-28} ${09:34}

@Desc    :   practice

'''

import os
import torch
#需要导入mobilenet模块，不然在终端运行demo的时候会报错说，找不到MobileNet模块
import sys
sys.path.append('..')
from MobileNet.mobileNet import *
sys.path.append('./MobileNet')

import argparse
from PIL import Image
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt


parse = argparse.ArgumentParser('Choose mobilenet version')
parse.add_argument('-v','--version',default='V2',type = str, help= 'input V1, V2 or V3 please')
args = parse.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = torch.load('./model/MobileNet%s.pkl'%args.version)
print('\nThe test will use MobileNet%s'%args.version)
summary(model,(3,224,224))
model.to(device)
model.eval()

dataloader = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()])
classes = ['A', 'B', 'C', 'D', 'E']

src = '../data/asl_dataset/valid'
files = os.listdir(src)
for file in files:
    path = os.path.join(src,file)
    ims = os.listdir(path)
    for i,im in enumerate(ims):
        row_data = Image.open(os.path.join(path,im))
        data2tensor = dataloader(row_data)
        data2tensor.unsqueeze_(0)
        data2tensor = data2tensor.to(device)
        out = model(data2tensor)
        pred = out.max(1,keepdim = True)[1].item()

        plt.ion()#使用终端时打开，使用IDE时注释掉
        plt.figure()
        plt.text(10, 10, ('No %d/%d' % (i, 600) ), fontdict={'size': 15, 'color': 'white'})
        plt.text(10, 22, ('real  :'+file),fontdict = {'size':15,'color':'red'})
        plt.text(10, 34, ('pred :'+str(classes[pred])),fontdict = {'size':15,'color':'blue'} )
        plt.imshow(row_data)
        # plt.show()#使用IDE时打开，使用终端时注释掉
        plt.pause(0.5)
        plt.close('all')

















