import torch
import torch.nn as nn

"""
Deeplab v3：Rethinking Atrous Convolution for Semantic Image Segmentation
Deeplab v3+：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

DeepLavV3+论文 2. Related Work 里Encoder-Decoder说采用DeepLavV3作为encoder module，
在此基础上，增加decoder module。我们看一下这个encoder module怎么做Atrous Conv的。
DeepLabV3论文 Fig3 展示了如何使用Atros Conv，以ResNet101为例。
Conv1 + MaxPool（包括layer1) 得到4x的output，正常操作。
layer2 得到8x的output，正常操作。
layer3 得到16x的output，正常操作。
layer4不同了，采用了Atrous Conv，不再下采样。也就是说dilation=2，stride=1
"""


def conv_3x3(in_channels, out_channels, stride=1, dilation=1):
    """
    3x3 same 卷积
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param stride: 下采样率。默认stride=1，不下采样；stride=2，下采样2倍
    :param dilation: Atros Conv的rate
    :return: 3x3 same 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)  # 后面接bn，bias=False;same卷积padding=1


def conv_1x1(in_channels, out_channels, stride=1):
    """
    用于调整维度
    layer调整channel
    project调整channel，spatial
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :param stride: 下采样率。默认stride=1，不下采样；stride=2，下采样2倍
    :return: 1x1 卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False)  # 后面接bn，bias=False


class BasicBlock(nn.Module):
    expansion = 1  # Basic Block 最后一个卷积输出channels是plane的1倍

    def __init__(self, inplanes, planes, stride=1, dilation=1, batch_norm=None,
                 project=None):
        """
        Basic Block 每个block有2个3x3卷积，两个卷积的输出通道数相同，等于planes。
        :param inplanes: 这个basic block的输入通道数，前一个basic block输出通道数。
        :param planes: 两个卷积的输出通道数相同，等于planes。取值64,128,256,512。
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :param batch_norm: 外部指定bn，不指定就用默认bn。
        :param project: 外部指定project也就是残差中的+x方法。
        """
        super(BasicBlock, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 外部不指定bn就使用默认bn

        # 第一个3x3卷积，论文中conv2_x不下采样，conv3_x-conv5_x下采样
        self.conv1 = conv_3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3x3卷积，都不下采样
        self.conv2 = conv_3x3(planes, planes, dilation=dilation)
        self.bn2 = batch_norm(planes)

        # 维数一致才能相加， +x 或者 +project(x)
        self.project = project
        pass

    def forward(self, x):
        identity = x  # 记录下输入x

        # 第一个卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 维数一致才能相加
        # 判断+identity，还是+project(x)
        if self.project is not None:
            identity = self.project(x)
        out += identity  # 残差+
        out = self.relu(out)  # 加上以后再relu
        return out

    pass


class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck最后一个卷积输出channels是planes的4倍

    def __init__(self, inplanes, planes, stride=1, dilation=1, batch_norm=None,
                 project=None):
        """
        Bottleneck
        每个block有3个卷积：1x1降低channel；3x3卷积下采样或不下采样；1x1升高channels；减小计算量。
        前两个卷积的输出channels相同，都等于planes
        最后一个卷积的输出channels是planes的4倍
        :param inplanes: 这个basic block的输入通道数，前一个basic block输出通道数。
        :param planes: 前两个卷积的输出通道数相同，等于planes。取值64,128,256,512。
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :param batch_norm: 外部指定bn，不指定就用默认bn。
        :param project: 外部指定project也就是残差中的+x方法。
        """
        super(Bottleneck, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 外部不指定bn就使用默认bn

        # 第一个1x1卷积，降低channels
        self.conv1 = conv_1x1(inplanes, planes)
        self.bn1 = batch_norm(planes)
        self.relu = nn.ReLU(inplace=True)

        # 第二个3x3卷积，下采样或不下采样
        self.conv2 = conv_3x3(planes, planes, stride=stride, dilation=dilation)
        self.bn2 = batch_norm(planes)

        # 第三个1x1卷积，升高channels到planes的4倍
        self.conv3 = conv_1x1(planes, self.expansion * planes)
        self.bn3 = batch_norm(self.expansion * planes)

        # 维数一致才能相加，+x 或者 +project(x)
        self.project = project
        pass

    def forward(self, x):
        identity = x  # 记录下输入x

        # 第一个1x1卷积，降低channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个3x3卷积，下采样或不下采样
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个1x1卷积，升高channels到planes的4倍
        out = self.conv3(out)
        out = self.bn3(out)

        # 维数一致才能相加
        # +x 或者 +project(x)
        if self.project is not None:
            identity = self.project(x)
        out += identity  # 残差+
        out = self.relu(out)  # 加上以后再relu
        return out

    pass


class ResNetBackBone(nn.Module):
    def __init__(self, block, layers, in_channels=3, batch_norm=None):
        """
        ResNet 18/34/50/101/152
        :param block: ResNet 18/34 用Basic Block
                      ResNet 50/101/152 用Bottleneck
        :param layers: 每种ResNet各个layer中block的数量。
                       取列表前4个数字，依次代表论文Conv2_x至Conv5_x中block的数量
        :param in_channels: 模型输入默认是3通道的
        :param batch_norm: 外部指定bn
        """
        super(ResNetBackBone, self).__init__()
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d  # 没有外部指定bn就用默认bn
        self._batch_norm = batch_norm

        self.inplanes = 64  # 各个layer输出通道数，conv1输出64通道，后面再make_layer中更新

        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)  # 后面接bn不要bias
        self.bn1 = self._batch_norm(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=1, dilation=2)

        self.layer5 = self._make_layer(block, layers[3], 512, stride=1,
                                       dilation=2)  # DeepLabV3中3.2. Going Deeper with Atrous Convolution
        self.layer6 = self._make_layer(block, layers[3], 512, stride=1, dilation=2)
        pass

    def _make_layer(self, block, n_block, planes, stride=1, dilation=1):
        """
        构造layer1-layer3，也就是论文中的Conv2_x-Conv4_x
        :param block: ResNet 18/34 用Basic Block
                      ResNet 50/101/152 用Bottleneck
        :param n_block: 本层block数量
        :param planes: 本层的基准channels数
        :param stride: stride=1，不下采样；
                       stride=2，第一个卷积下采样；
        :return:
        """
        batch_norm = self._batch_norm

        # 第一个block考虑设置project
        project = None
        if stride != 1 or self.inplanes != block.expansion * planes:
            # stride!=1 下采样，调整spatial
            # block输入channels和输出channels不一致，调整channels=block.expansion * planes
            project = nn.Sequential(
                conv_1x1(self.inplanes, block.expansion * planes,
                         stride=stride),
                batch_norm(block.expansion * planes)  # 调整维数后bn
            )

        # 第一个block考虑是否下采样，单独设置
        layer = [block(self.inplanes, planes, stride=stride, dilation=dilation,
                       batch_norm=batch_norm, project=project)]

        self.inplanes = block.expansion * planes  # 后面几个block输入channel

        # 其余block一样，都不进行下采样，循环添加
        for _ in range(1, n_block):  # 第一个block单独设置了，所以range从1开始
            layer.append(block(self.inplanes, planes, dilation=dilation,
                               batch_norm=batch_norm))
            pass

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.maxpool(x)  # 1/4
        x = self.layer1(x)
        low_level_features = x

        x = self.layer2(x)  # 1/8
        x = self.layer3(x)  # 1/16
        x = self.layer4(x)  # 1/16
        x = self.layer5(x)  # 1/16
        x = self.layer6(x)  # 1/16

        return x, low_level_features  # 输出主干特征x和low-level特征

    pass


def resnet50_atrous(in_channels=3, batch_norm=None):
    return ResNetBackBone(Bottleneck, [3, 4, 6, 3], in_channels=in_channels,
                          batch_norm=batch_norm)


def resnet101_atrous(in_channels=3, batch_norm=None):
    return ResNetBackBone(Bottleneck, [3, 4, 23, 3], in_channels=in_channels,
                          batch_norm=batch_norm)


if __name__ == '__main__':
    net = resnet50_atrous()
    # net = resnet101_backbone()
    print(net)

    in_data = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.float)
    print('in data:', in_data.shape)

    atrous_conv, low_level = net(in_data)
    print('atrous conv:', atrous_conv.shape)
    print('low-level:', low_level.shape)
