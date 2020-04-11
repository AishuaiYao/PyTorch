'''
@Author  :   {AishuaiYao}
@License :   (C) Copyright 2013-2020, {None}
@Contact :   {aishuaiyao@163.com}
@Software:   ${segmentation}
@File    :   ${voc}.py
@Time    :   ${2020-04-04}
@Desc    :   deconvlution experiment
'''

import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from FCN.fcn import *
import numpy as np
import cv2
from PIL import Image


classes = ['background',    'aeroplane',    'bicycle',      'bird',         'boat',
           'bottle',        'bus',          'car',          'cat',          'chair',
           'cow',           'diningtable',  'dog',          'horse',        'motorbike',
           'person',        'potted plant', 'sheep',        'sofa',         'train',
           'tv/monitor']

# RGB color for each class
colormap = [[0,0,0],        [128,0,0],      [0,128,0],      [128,128,0],    [0,0,128],
            [128,0,128],    [0,128,128],    [128,128,128],  [64,0,0],       [192,0,0],
            [64,128,0],     [192,128,0],    [64,0,128],     [192,0,128],    [64,128,128],
            [192,128,128],  [0,64,0],       [128,64,0],     [0,192,0],      [128,192,0],
            [0,64,128]]
voc_path = '../data/VOC2012'
BATCH_SIZE = 20
num_classes = 21
epochs = 500
input_size = 512


def read_images(path = voc_path, train = True):
    file = path + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(file) as f:
        imgs = f.read().split()
    datas = [path + '/JPEGImages/%s.jpg'%img for img in imgs]
    labels = [path + '/SegmentationClass/%s.png'%img for img in imgs]
    return datas, labels


def preproccessing(datas,labels):
    for i,img, label in enumerate(zip(datas, labels)):
        img_canvas,label_canvas = img2label(img,label)



def img2label(img,label,canvas_size = input_size):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    label = cv2.imread(label)
    label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)

    max_width, max_height = canvas_size,canvas_size
    height, width, channel = img.shape
    pad_width = (max_width - width) // 2
    pad_height = (max_height - height) // 2

    img_canvas = np.full((max_width, max_height, 3), 0)
    label_canvas = np.full((max_width, max_height, 3), 0)
    img_canvas[pad_height: pad_height + height, pad_width: pad_width + width, :] = img
    label_canvas[pad_height: pad_height + height, pad_width: pad_width + width, :] = label

    transform = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img2tensor = transform(img_canvas)

    for i, cm in enumerate(colormap):
        label_canvas[
            (label_canvas[:, :, 0] == cm[0]) & (label_canvas[:, :, 1] == cm[1]) & (label_canvas[:, :, 2] == cm[2])] = i
    label_canvas = label_canvas[:, :, 0]
    label_canvas[label_canvas == 224] = 0
    label2tensor = torch.from_numpy(label_canvas)

    return img2tensor, label2tensor


class VOCSegGenerator(Dataset):
    def __init__(self,train,):
        super(VOCSegGenerator, self).__init__()
        self.data_list, self.label_list = read_images(path=voc_path, train = train)
        self.len = len(self.data_list)
        print('Read '+ str(self.len)+' images')

    def __getitem__(self,idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img,label = img2label(img, label)
        return img,label

    def __len__(self):
        return self.len


train = VOCSegGenerator(train = True)
valid = VOCSegGenerator(train = False)

train_loader = DataLoader(dataset = train,batch_size=BATCH_SIZE,shuffle=True)
valid_loader = DataLoader(dataset = valid,batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN(num_classes)

# model.to(device)

device_ids = [2,3]
model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
model = model.cuda(device=device_ids[0])  # 模型放在主设备

summary(model,(3,input_size,input_size))
# model = torch.load('./model/fcn1.pkl')
# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=5*1e-4,momentum=0.9)


def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,label) in enumerate(train_loader):
        # data,label = data.to(device),label.to(device)
        data, label = data.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
        optimizer.zero_grad()
        output = model(data)

        output = F.log_softmax(output,dim=1)
        criterion = nn.NLLLoss()
        loss = criterion(output,label)

        loss.backward()
        optimizer.step()
        if (batch_idx) % 30 == 0:
            print('train {} epoch : {}/{} \t loss : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def predict(mdoel,device,valid_loader):
    model.eval()
    cnt = 0
    with torch.no_grad():
        for batch_idx,(data,label) in enumerate(valid_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)

            data = data.squeeze().cpu().numpy().transpose((1, 2, 0)) * 255
            data = data[:,:,::-1]
            pred = output.max(dim=1)[1].squeeze().cpu().numpy()
            cm = np.array(colormap).astype('uint8')
            label = label.squeeze().cpu().numpy()
            label = cm[label][:,:,::-1]#becuse opencv channel is bgr
            pred = cm[pred][:,:,::-1]

            cv2.imwrite('./result/%d_img.jpg' % batch_idx, data)
            cv2.imwrite('./result/%d_label.jpg'%batch_idx,label)
            cv2.imwrite('./result/%d_pred.png'%batch_idx,pred)
            cnt+=1
            if cnt>6:
                break


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(epochs):
    print(epoch)
    if epoch < 180:
        adjust_learning_rate(optimizer,epoch)
    train(model,device,train_loader,optimizer,epoch)
    torch.save(model,'./model/fcn0.pkl')

# predict(model,device,valid_loader)
