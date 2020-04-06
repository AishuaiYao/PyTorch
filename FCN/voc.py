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
import time
from PIL import Image
import torch
from torchvision import transforms,datasets
from torch.utils.data import Dataset,DataLoader
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





def img2label(img,label):
    img = cv2.imread(img)
    label = cv2.imread(label)

    max_width, max_height = 500, 500
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

train_loader = DataLoader(dataset = train,batch_size=16,shuffle=True)
vaild_loader = DataLoader(dataset = valid,batch_size=16)


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


import torch.nn as nn

# from mxtorch.trainer import ScheduledOptim

from FCN import fcn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fcn.FCN(num_classes=21)
model.to(device)

criterion = nn.NLLLoss2d()
basic_optim = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
# optimizer = ScheduledOptim(basic_optim)
