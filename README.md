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
* YOLOV3
## 数据集
* mnist
* asl，A-E共五类，共15000个数据，12000的训练集、3000的验证集
## 使用方法
### 使用pycharm
### 使用终端
#### 在MobileNet目录下
* 训练模型：
> python asl.py -v V2 -e 20  
> python asl.py -v V3 -t large -e 20
* demo演示: 
> python demo.py -v V2

#### 在YOLOV3目录下   
* 获取yolov3的权重文件   
> wget https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png
* 将待测试图片放入test文件夹，输出结果保存在detect文件夹中  
* 执行  
> python demo.py
* 示例  
![image](https://github.com/AishuaiYao/PyTorch/blob/master/YOLOV3/detect/person.jpg)

#### 在FCN目录下   
* 需要提前下载好VOC2012数据集，并按程序中的规定的目录地址放在data文件夹中
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 实验了vgg和resnet两种不同backbone的fcn结构，其余所有策略包括数据的形式和优化策略都相同。迭代160次，需要高性能GPU支持。实验结果中resnet50基础的网络最接近论文效果  
* 从左到右依次是：原图 标签 FCNx8_ResNet50 FCNx8_VGG  
![image](https://github.com/AishuaiYao/PyTorch/blob/master/FCN/test/39759931.jpg)  
* 结论
  1. 网络的效果因结构的不同有很大差异，网络结构很重要  
  2. 在实验中两种网络的最小损失都只能降低到小数点后两位0.0xx停滞  
  3. Adam不能降低损失，按照论文中的SGD配置效果最好  



