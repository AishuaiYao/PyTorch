'''
@Author  :   {AishuaiYao}

@License :   (C) Copyright 2013-2020,{None}

@Contact :   {aishuaiyao@163.com}

@Software:   ${yolov3 experiment}

@File    :   ${darknet}.py

@Time    :   ${2020-03-03}

@Desc    :   practice

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from YOLOV3.util import *


def cfg_parse(cfg):

    block = {}
    blocks = []
    with open(cfg, 'r') as file:
        lines = file.read().split('\n')
    lines= [i for i in lines if len(i) > 0 and i[0] !='#']
    for line in lines:
        if line[0] == '[':
            if len(block) > 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1: -1]
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks



def build_net(blocks):
    net_info = blocks[0]
    modules = nn.ModuleList()
    in_channels = 3
    out_channels = 0
    channel_record = []

    for i,block in enumerate(blocks[1:]):

        module = nn.Sequential()

        if block['type'] == 'convolutional':

            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            out_channels = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            activation = block['activation']

            padding = (kernel_size - 1) // 2
            conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=bias)
            module.add_module('convolutional_{}'.format(i),conv)
            if batch_normalize == 1:
                bn = nn.BatchNorm2d(out_channels)
                module.add_module('batch_normalize_{}'.format(i),bn)
            if activation == 'leaky':
                act = nn.LeakyReLU(0.1,inplace=True)
                module.add_module('leaky_{}'.format(i),act)



        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=stride,mode = 'nearest')
            module.add_module('upsample_{}'.format(i),upsample)

        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = list(map(int,layers))
            # layers = list(map(lambda x: x-i if x>0 else x,layers))
            try:
                start, end = layers[0], layers[1]
                end = end - i
            except:
                start, end = layers[0], 0

            route = EmptyLayer()
            module.add_module('route_{}'.format(i),route)

            if end < 0:
                out_channels = channel_record[i + start] + channel_record[i + end]
            else:
                out_channels = channel_record[i + start]

        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(i),shortcut)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = list(map(int,mask))

            anchors = block['anchors'].split(',')
            anchors = list(map(int,anchors))

            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('yolo_{}'.format(i),detection)

        modules.append(module)
        in_channels = out_channels
        channel_record.append(out_channels)

    return (net_info,modules)

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()



class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



class DarkNet(nn.Module):
    def __init__(self,cfg):
        super(DarkNet, self).__init__()
        self.blocks = cfg_parse(cfg)
        self.net_info, self.modules_list = build_net(self.blocks)


    def forward(self,x, cuda):
        feature_map = []
        detection = None

        for i,block in enumerate(self.blocks[1:]):
            if block['type'] == 'convolutional' or block['type'] == 'upsample':
                x = self.modules_list[i](x)

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = list(map(int,layers))

                if layers[0] > 0 :
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = feature_map[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] -i
                    map1 = feature_map[i + layers[0]]
                    map2 = feature_map[i + layers[1]]
                    x = torch.cat((map1,map2),1)

            elif block['type'] == 'shortcut':
                front = int(block['from'])
                x = feature_map[i-1]+feature_map[i+front]

            elif block['type'] == 'yolo':
                anchors =self.modules_list[i][0].anchors
                height = int(self.net_info['height'])
                num_classes = int(block['classes'])

                x = x.data
                x = predict_transform(x,height,anchors,num_classes,cuda)

                if detection is not None:
                    detection = torch.cat((detection, x), 1)
                else:
                    detection = x

            feature_map.append(x)

        return detection

    def load_weight(self,path = './model/yolov3.weights'):
        with open(path, 'rb') as file:
            header = np.fromfile(file,dtype = np.int32,count = 5)
            self.header= torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(file, dtype=np.float32)
            ptr = 0
            for i in range(len(self.modules_list)):
                module_type = self.blocks[i+1]['type']
                if module_type == 'convolutional':
                    model = self.modules_list[i]
                    try:
                        batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                    except:
                        batch_normalize = 0

                    conv = model[0]
                    if batch_normalize:
                        bn = model[1]

                        num_bn_biases = bn.bias.numel()

                        bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr+=num_bn_biases

                        bn_weight = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases


                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weight = bn_weight.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        bn.bias.data.copy_(bn_biases)#将src中的元素复制到tensor中并返回这个tensor; 两个tensor应该有相同shape
                        bn.weight.data.copy_(bn_weight)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        num_biases = conv.bias.numel()
                        conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                        ptr += num_biases
                        conv_biases = conv_biases.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_biases)

                    num_weight = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weight])
                    ptr += num_weight

                    conv_weights = conv_weights.view_as(conv.weight.data)

                    conv.weight.data.copy_(conv_weights)







#
# import cv2
# from torch.autograd import Variable
#
# def get_test_input():
#     img = cv2.imread("./dog-cycle-car.png")
#     img = cv2.resize(img, (416,416))          #Resize to the input dimension
#     img_ = img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
#     img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
#     img_ = torch.from_numpy(img_).float()     #Convert to float
#     img_ = Variable(img_)                     # Convert to Variable
#     return img_
# model = DarkNet("./cfg/yolov3.cfg")
# model.load_weight('./model/yolov3.weights')
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print (pred)
#
#
