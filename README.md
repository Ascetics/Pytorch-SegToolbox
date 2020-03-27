# Pytorch-SegToolbox
用Pytorch复现常用语义分割网络，备用

# 目前实现DeepLabV3+、UNet

# 目录结构
Pytorch-SegToolbox   
&nbsp;&nbsp;|--backbones 存放主干网络ResNet、AlignedXception等  
&nbsp;&nbsp;|--datasets 存放数据集、观察数据工具等  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--data_list 存放数据集的csv文件  
&nbsp;&nbsp;|--models 存放语义分割网络 DeepLabV3+（Xception|ResNet101）、UNet（ResNet）、FCN8s等  
&nbsp;&nbsp;|--res 存放资源 权重文件、图像、日志等  
&nbsp;&nbsp;|--utils 存放工具  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--augment数据增强  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--lossfn损失函数  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--tools日志、计时、评价标准metrics等工具  
&nbsp;&nbsp;|--test.py 用来测试  
&nbsp;&nbsp;|--train.py 用来训练  

# 权重文件&测试文件
由于训练资源（租用GPU）太烧钱，只训练了DeepLabV3+（Xception）和FCN8s。
权重文件，以及以DeepLabV3+（Xception）的测试输出的图片可以在此处下载:  
链接：https://pan.baidu.com/s/1U0lx83uKexpfYPaTen3vog  
提取码：iyqd   
res下的preds只展示了几张效果不错的测试输出，效果不好的，我就不贴了。哈哈……


