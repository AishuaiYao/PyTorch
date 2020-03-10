'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2020,{None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${yolov3 experiment}

@File    :   ${util}.py

@Time    :   ${2020-03-05}

@Desc    :   practice

'''
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np




def predict_transform(x,in_dim,anchors,num_classes,cuda):
    batch_size,channel,height,width, = x.size()
    stride = in_dim // height
    unit = 5+num_classes
    num_anchors = len(anchors)

    x = x.view(batch_size,unit * num_anchors,height * width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size,(height * width)*num_anchors,unit)

    anchors = [(anchor[0]/stride ,anchor[1]/stride) for anchor in anchors]

    x[:,:,0] = torch.sigmoid(x[:,:,0])
    x[:,:,1] = torch.sigmoid(x[:,:,1])
    x[:,:,4] = torch.sigmoid(x[:,:,4])

    grid = np.arange(height)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if cuda :
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    offset = torch.cat((x_offset,y_offset),1)
    offset = offset.repeat(1,num_anchors)
    offset = offset.view(-1,2)
    offset = offset.unsqueeze(0)
    x[:,:,:2] += offset#针对cell为1的转换


    anchors = torch.FloatTensor(anchors)

    if cuda :
        anchors = anchors.cuda()

    anchors = anchors.repeat(height * width,1).unsqueeze(0)
    x[:, :, 2:4] = torch.exp(x[:,:,2:4]) * anchors
    x[:, :, 5:5+num_classes] = torch.sigmoid((x[:,:,5:5+num_classes]))
    x[:,:,:4] *=stride

    return x


def reprocessing(model_output, confidence, num_classes, nms_threshold=0.5):
    result = []
    for x in model_output:
        mask = (x[:,4] > confidence).float().unsqueeze(1)
        if mask.shape[0] == 0:
            return 0
        x *= mask
        selected = (torch.nonzero(x[:, 4]))
        x = x[selected.squeeze(), :]

        try:
            score, classes = torch.max(x[:,5:],1)
        except:
            return 0
        classes = classes.float().unsqueeze(1)
        score = score.float().unsqueeze(1)
        x = torch.cat((x[:,:5],classes,score),1)

        box_corner = x.new(x.shape)
        box_corner[:, 0] = (x[:, 0] - x[:, 2] / 2)
        box_corner[:, 1] = (x[:, 1] - x[:, 3] / 2)
        box_corner[:, 2] = (x[:, 0] + x[:, 2] / 2)
        box_corner[:, 3] = (x[:, 1] + x[:, 3] / 2)
        x[:, :4] = box_corner[:, :4]

        classes = unique(x[:,-2])

        pred = []
        for cls in classes:
            mask = (x[:, -2] == cls)
            selected = torch.nonzero(mask)
            bboxs = x[selected.squeeze(), :]
            target = nms(bboxs, nms_threshold)
            pred.append(target)
        result.append(pred)

    return result


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    return unique_np

def bbox_iou(box1, box2):
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    l = torch.max(box1[0],box2[0])
    r = torch.min(box1[2],box2[2])
    t = torch.max(box1[1],box2[1])
    b = torch.min(box1[3],box2[3])

    if l>=r or t>=b:
        return 0
    else:
        area = (r-l)*(b-t)
        return area/(area1+area2-area)


def nms(bboxs, thred):
    result = []
    while bboxs.size(0):
        try:
            score = bboxs[:, -1]
        except:
            bboxs = bboxs.unsqueeze(0)
            score = bboxs[:, -1]
        _, idx = torch.max(score, 0)
        best = bboxs[idx]
        result.append(best)
        bboxs = torch.cat((bboxs[:idx],bboxs[idx+1:]),0)
        record = []
        for i,bbox in enumerate(bboxs):
            iou = bbox_iou(best,bbox)
            if iou<thred:
                record.append(i)
        record = torch.from_numpy(np.array(record))
        try:
            bboxs = bboxs[record,:]
        except:
            break
    return result


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim

    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img




def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
























