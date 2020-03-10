from __future__ import division
import numpy as np
import cv2
import os
import argparse
import sys
sys.path.append('..')
from YOLOV3.util import *
from YOLOV3.darknet import DarkNet

sys.path.append('./MobileNet')



def parse_arg():
    parser  = argparse.ArgumentParser('args of YOLOV3 model')
    parser.add_argument('--images',default='./test', type = str, help = 'input images directory')
    parser.add_argument('--in_dim', default= 416, type=int, help='model input dim')
    return  parser.parse_args()

def load_data(args):
    files = os.listdir(args.images)
    raw_imgs = [cv2.imread(os.path.join(args.images,file))  for file in files]
    imgs_dim = [img.shape  for img in raw_imgs]
    new_imgs = list(map(prep_image,raw_imgs, [args.in_dim for i in range(len(raw_imgs))]))

    return imgs_dim,raw_imgs,new_imgs,files

def load_model(args,CUDA):
    model = DarkNet('./cfg/yolov3.cfg')
    model.load_weight('./model/yolov3.weights')
    model.net_info['height'] = args.in_dim

    if CUDA:
        model.cuda()

    model.eval()

    return model

def save_result(args,result,img,imgs_dim,file_name):
    for targets in result[0]:
        for target in targets:
            target = list(target.cpu().numpy())

            scale = min(args.in_dim / imgs_dim[0], args.in_dim / imgs_dim[1])
            new_h = int(imgs_dim[0] * scale)
            new_w = int(imgs_dim[1] * scale)


            zero_padding_h = (args.in_dim - new_h) // 2
            zero_padding_w = (args.in_dim - new_w) // 2

            target[0] = int((target[0] - zero_padding_w) / scale)
            target[1] = int((target[1] - zero_padding_h) / scale)
            target[2] = int((target[2] - zero_padding_w) / scale)
            target[3] = int((target[3] - zero_padding_h) / scale)

            classes = load_classes('./cfg/coco.names')

            cv2.rectangle(img, (target[0], target[1]), (target[2], target[3]), color=(0, 255, 255), thickness=2)
            cv2.putText(img,classes[int(target[-2])], (target[0]+2, target[1]+15),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,color = (255,0,255),thickness=2)
            cv2.putText(img,'score:{:.2}'.format(target[-1]), (target[0]+2, target[1] + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5, color=(255, 0,255), thickness=2)
            cv2.imwrite('./detect/{}'.format(file_name),img)

if __name__=='__main__':
    CUDA = torch.cuda.is_available()
    args = parse_arg()
    imgs_dim,raw_imgs, new_imgs,files = load_data(args)
    yolov3 = load_model(args,CUDA)
    for i,img in enumerate(new_imgs):
        if CUDA:
            img = img.cuda()
        with torch.no_grad():
            output = yolov3(Variable(img),CUDA)
        result = reprocessing(output,confidence=0.5,num_classes=80,nms_threshold=0.3)
        if result:
            save_result(args,result,raw_imgs[i],imgs_dim[i],files[i])
        else:
            print()
            print(files[i]+' have no target')



    print('complete')