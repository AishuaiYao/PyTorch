# PyTorchPracticeProject
## 环境说明
* torch1.0
* torchvision0.2.0
* python3.5
* ubuntu16.04
* pycharm2018
## 搭建网络
* ResNet
* MobileNet
## 数据集
* mnist
* asl，A-E共五类，共15000个数据，12000的训练集、3000的验证集
## 使用方法
### 使用pycharm
### 使用终端
在MobileNet目录下
* 训练模型：
> python asl.py -v V2 -e 20  
> python asl.py -v V3 -t large -e 20
* demo演示: 
> python demo.py -v V2
在YOLOV3目录下   
* 获取yolov3的权重文件   
> wget https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png
* 将待测试图片放入test文件夹，输出结果保存在detect文件夹中  
* 执行  
> python demo.py
* 示例  
![image](https://github.com/AishuaiYao/PyTorch/blob/master/YOLOV3/detect/person.jpg)


