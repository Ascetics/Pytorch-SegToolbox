import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modified Aligned Xception

(1) deeper Xception same as in [31] except that we do not modify the entry flow 
network structure for fast computation and memory efficiency

更深的网路， Middle Flow 重复16个block

(2) all max pooling operations are replaced by depthwise separable
convolution with striding, which enables us to apply atrous separable convolu-
tion to extract feature maps at an arbitrary resolution (another option is to
extend the atrous algorithm to max pooling operations)

将Maxpool下采样改为带有stride的深度可分离卷积SepConv2d

(3) extra batch normalization [75] and ReLU activation are added after each 3x3 
depthwise convolution, similar to MobileNet design [29]



"""


def _print_shape(func):
    """
    测试打印输入输出维度
    :param func:
    :return:
    """

    def print_shape(*args, **kwargs):
        res = func(*args, **kwargs)
        print(func, res.shape)
        return res

    return print_shape


class SepConv2d(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1,
                 dilation=1, bias=False):
        """
        深度可分离卷积
        第一个卷积depthwise_conv在spatial上，每个channel单独进行，用group=in_planes
        第二个卷积pointwise_conv在cross-channel上，相当于用1x1卷积调整维度
        :param in_planes: in_channels
        :param planes: out_channels
        :param kernel_size: kernel_size
        :param stride: depthwise conv的stride
        :param dilation: depthwise conv的padding
        :param bias: bias，因为后面都接BN，默认False
        """
        super(SepConv2d, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation  # 都是same卷积
        self.depthwise_conv = nn.Conv2d(
            in_planes, in_planes, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_planes, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_planes, planes, 1, bias=bias)
        pass

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

    pass


class Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dilation=1):
        """
        Entry Flow和Middlw Flow用的Block
        只有第三个卷积来决定是否下采样
        :param in_planes: in_planes
        :param planes: planes
        :param stride: stride决定第三个卷积是否下采样
        :param dilation: dilation
        """
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(in_planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=stride, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))  # 第三个conv的stride决定是否下采样
        self.project = None
        if in_planes != planes or stride != 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride),
                nn.BatchNorm2d(planes))  # residual connection
        pass

    # @_print_shape
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.project is not None:
            identity = self.project(identity)  # residual connection 相加
        x = x + identity
        return F.relu(x, inplace=True)

    pass


class ExitBlock(nn.Module):
    def __init__(self, in_planes=728, planes=1024, stride=1, dilation=1):
        """
        Exit Flow用的Block
        728->728
        728->1024
        1024->1024
        :param in_planes: in_planes=728
        :param planes: planes=1024
        :param stride: stride决定第三个卷积是否下采样
        :param dilation: dilation
        """
        super(ExitBlock, self).__init__()
        self.conv1 = nn.Sequential(
            SepConv2d(in_planes, in_planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            SepConv2d(in_planes, planes, 3, stride=1, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            SepConv2d(planes, planes, 3, stride=stride, dilation=dilation),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))  # 第三个conv的stride决定是否下采样
        self.project = None
        if in_planes != planes or stride != 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride),
                nn.BatchNorm2d(planes))
        pass

    # @_print_shape
    def forward(self, x):
        identity = x  # residual connection 准备
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.project is not None:
            identity = self.project(identity)  # residual connection 相加
        x = x + identity
        return x

    pass


class XceptionBackbone(nn.Module):
    def __init__(self, in_planes, output_stride=16):
        """
        用于DeepLabV3+的AlignedXception
        :param in_planes: 输入通道，也就是图像的通道
        :param output_stride: 主干输出spatial与输入spatial的比值可以是8,16,32,论文采用16最好
        """
        super(XceptionBackbone, self).__init__()
        # 根据DeepLabV3讨论的Atrous Conv
        # strides[0]和dilations[0] 在Entry Flow的第三个block使用
        # Middle Flow都不进行下采样，stride都等于1，dilation使用dilations[1]
        # strides[1]和dilations[1] 在Exit Flow的block使用
        if output_stride == 8:  # os=8时，最后一次下采样dilation=1，之后dilation=4
            strides = (1, 1)
            dilations = (4, 4)
        elif output_stride == 16:  # os=16时，最后一次下采样dilation=1，之后dilation=2
            strides = (2, 1)
            dilations = (1, 2)
        elif output_stride == 32:  # os=32时，最后一次下采样dilation=1，之后dilation=1
            strides = (2, 2)
            dilations = (1, 1)
        else:
            raise ValueError('output stride error!')

        # Entry Flow
        self.entry_conv1 = nn.Sequential(
            nn.Conv2d(in_planes, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.entry_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.entry_block1 = nn.Sequential(Block(64, 128, stride=2))
        self.entry_block2 = nn.Sequential(Block(128, 256, stride=2))
        self.entry_block3 = nn.Sequential(
            Block(256, 728, stride=strides[0], dilation=dilations[0]))

        # Middle Flow
        mid_blocks = [Block(728, 728, stride=1, dilation=dilations[1])] * 16
        self.mid_blocks = nn.Sequential(*mid_blocks)

        # Exit Flow
        self.exit_block = ExitBlock(728, 1024, stride=strides[1], dilation=dilations[1])
        self.exit_conv1 = nn.Sequential(
            SepConv2d(1024, 1536, 3, dilation=dilations[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True))
        self.exit_conv2 = nn.Sequential(
            SepConv2d(1536, 1536, 3, dilation=dilations[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True))
        self.exit_conv3 = nn.Sequential(
            SepConv2d(1536, 2048, 3, dilation=dilations[1]),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True))
        self._init_weight()
        pass

    def _init_weight(self):
        """
        初始化参数
        :return:
        """
        pass

    def forward(self, x):
        # Entry Flow
        x = self.entry_conv1(x)  # 2x
        x = self.entry_conv2(x)  # 2x
        x = self.entry_block1(x)  # 4x
        low_level_features = x  # low-level feature
        x = self.entry_block2(x)  # 8x
        x = self.entry_block3(x)  # os=8,8x|os=16,16x|os=32,16x

        # Middle Flow
        x = self.mid_blocks(x)  # os=8,8x|os=16,16x|os=32,16x

        # Exit Flow
        x = self.exit_block(x)  # os=8,8x|os=16,16x|os=32,32x 此后不再下采样
        x = self.exit_conv1(x)
        x = self.exit_conv2(x)
        x = self.exit_conv3(x)

        return x, low_level_features

    pass


def xception_backbone(in_channels, output_stride=16):
    if output_stride in (8, 16, 32):
        return XceptionBackbone(in_channels, output_stride)
    else:
        raise ValueError('output stride error! should be 8, 16 or 32')


if __name__ == '__main__':
    batch_size = 1
    in_dims = 3
    num_class = 8
    im = torch.randint(0, 256, size=(batch_size, in_dims, 299, 299),
                       dtype=torch.float, requires_grad=True)

    print(im.shape)

    model = xception_backbone(in_dims, 16)
    output, low_level = model(im)
    print(output.shape, low_level.shape)

    lb = torch.randint(0, num_class,
                       size=(output.shape[0], output.shape[2], output.shape[3]),
                       dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters())
    loss = F.cross_entropy(output, lb)
    loss.backward()
    optimizer.step()
    print(loss.detach().item())
    pass
