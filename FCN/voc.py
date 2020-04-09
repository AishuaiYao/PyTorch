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
from torchsummary import summary
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
from FCN.fcn import FCN
import numpy as np
import cv2


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


def read_images(path = voc_path, train = True):
    file = path + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(file) as f:
        imgs = f.read().split()
    datas = [path + '/JPEGImages/%s.jpg'%img for img in imgs]
    labels = [path + '/SegmentationClass/%s.png'%img for img in imgs]
    return datas, labels



def preproccessing(datas,labels):
    # width = [Image.open(i).size[0] for i in datas]
    # height = [Image.open(i).size[1] for i in datas]
    # max_width, max_height = 500, 500 #max(width), max(height)

    for img, label in zip(datas, labels):
        img2label(img,label)





def img2label(img,label,padding_size = 512):
    img = cv2.imread(img)
    # cv2.imshow('im',img)
    # cv2.waitKey(2000)
    label = cv2.imread(label)

    max_width, max_height = padding_size,padding_size
    height, width, channel = img.shape
    pad_width = (max_width - width) // 2
    pad_height = (max_height - height) // 2

    img_canvas = np.full((max_width, max_height, 3), 0)
    label_canvas = np.full((max_width, max_height, 3), 0)
    img_canvas[pad_height: pad_height + height, pad_width: pad_width + width, ::-1] = img
    label_canvas[pad_height: pad_height + height, pad_width: pad_width + width, ::-1] = label

    transform = transforms.Compose([transforms.ToTensor()])
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

Batch_size = 1
train_loader = DataLoader(dataset = train,batch_size=Batch_size,shuffle=True)
vaild_loader = DataLoader(dataset = valid,batch_size=Batch_size)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 21
model = FCN(num_classes)
model.to(device)

summary(model,(3,512,512))

optimizer = torch.optim.Adam(model.parameters())



def _fast_hist(label,pred,num_classes):
    mask = (label >= 0) &(label < num_classes)

    hist = np.bincount(num_classes * label[mask].astype(int) + pred[mask],minlength=num_classes**2).reshape(num_classes,num_classes)
    return hist

def label_accuracy_score(label,pred,num_classes):
    hist = np.zeros((num_classes,num_classes))
    for l,p in zip(label,pred):
        hist += _fast_hist(l.flatten(),p.flatten(),num_classes)
        # print(1)

    return 1,2,3,4

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,label) in enumerate(train_loader):
        data,label = data.to(device),label.to(device)
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
        pred = output.max(dim=1)[1].squeeze().data.cpu().numpy()
        # label = label.data.cpu().numpy()
        #
        # for t,p in zip(label,pred):
        #     acc,acc_cls,mean_iu,fwavac = label_accuracy_score(t,p,num_classes)


        cm = np.array(colormap).astype('uint8')
        pred = cm[pred]
        cv2.imwrite('./result/%d.png'%batch_idx,pred)



for epoch in range(1):
    train(model,device,train_loader,optimizer,epoch)

    print('1')













































