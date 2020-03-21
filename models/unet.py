import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


def unet_conv(in_channels, out_channels, padding=1):
    """
    UNet网络中block的基本实现，encode和decode类似，都是2个3x3卷积。
    与论文中的实现不同的是，默认加了padding都是same卷积
    :param in_channels: 输入channels
    :param out_channels: 输出channels
    :param padding: 默认加padding
    :return:
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=False),
                         # 第一个3x3卷积，后面接bn，bias=False
                         nn.BatchNorm2d(out_channels),  # bn
                         nn.ReLU(inplace=True),  # relu激活函数
                         nn.Conv2d(out_channels, out_channels, 3, padding=padding, bias=False),
                         # 第二个3x3卷积，后面接bn，bias=False
                         nn.BatchNorm2d(out_channels),  # bn
                         nn.ReLU(inplace=True))  # relu激活函数


class _UNetEncoder(nn.Module):
    def __init__(self, encode_blocks):
        """
        encoder部分。
        第一个encode block没有下采样。
        最后一个encode block没有shortcut。
        :param encode_blocks: encode每个层的block
        """
        super(_UNetEncoder, self).__init__()
        self.encoder = nn.ModuleList(encode_blocks)  # module保存所有的encode block
        pass

    def forward(self, x):
        shortcuts = []  # 保存所有的shortcut
        for encode in self.encoder:
            x = encode(x)  # 每个encode block的输出都保存
            shortcuts.append(x)
        return x, shortcuts[:-1]  # 返回提取的特征x，和所有shortcut。最后一个block输出不作为shortcut。

    pass


class _UNetDecoder(nn.Module):
    def __init__(self, encode_out_channels, n_class):
        """
        decoder部分。
        decode block比encode block少一个。
        所有的decode block都是上采样up -> 和shortcut拼接 -> decode操作。
        decode操作都是类似的unet_conv，是两个3x3卷积。与论文实现不同的是默认加padding使用same卷积。
        最后，增加一个1x1卷积，用于最后的分类。
        :param encode_out_channels: 列表类型，每个encode block的输出channels，按照encode block的顺序。
        :param n_class: n种分类。
        """
        super(_UNetDecoder, self).__init__()
        self.ups = nn.ModuleList()  # 上采样
        self.decodes = nn.ModuleList()  # decode

        in_channels = encode_out_channels[-1]  # 最后一个encode block的输出channels作为decode的输入channels
        for cat_channels in reversed(encode_out_channels[:-1]):  # decode与encode顺序相反，遍历所有剩余的encode block的输出channels
            out_channels = in_channels // 2  # 上采样输出channels是输入channels的一半,spatial增大一倍
            self.ups.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )

            in_channels = out_channels + cat_channels  # 与shortcut进行cat，改变了decode的输入channels
            # out_channels = in_channels // 2  # decode输出channels是输入channels的一半
            self.decodes.append(unet_conv(in_channels, out_channels))  # decode，decode是类似的都是2个3x3的same卷积

            in_channels = out_channels  # decode的输入channels作为下一次迭代的输入channels
            pass
        self.classifier = nn.Conv2d(in_channels, n_class, 1)  # 1x1卷积得到最终分类预测
        pass

    def forward(self, x, shortcuts):
        for i, (up, decode) in enumerate(zip(self.ups, self.decodes)):
            x = up(x)  # 先上采样
            x, s = self._crop(x, shortcuts[-i - 1])  # 剪裁大小，因为下采样上采样等会使x和shortcut的spatial大小不一致
            x = torch.cat([x, s], dim=1)  # 沿dim=1，也就是channel方向cat
            x = decode(x)  # decode，decode是类似的都是2个3x3的same卷积
        x = self.classifier(x)  # 1x1卷积得到最终分类预测
        return x

    @staticmethod
    def _crop(x, shortcut):
        """
        按照x和shortcut最小值剪裁
        :param x: 上采样结果
        :param shortcut: 就是shortcut
        :return: 剪裁后的x和shortcut
        """
        _, _, h_x, w_x = x.shape  # 取特征的spatial大小
        _, _, h_s, w_s = shortcut.shape  # 取shortcut的spatial大小
        h, w = min(h_x, h_s), min(w_x, w_s)  # 取最小spatial
        hc_x, wc_x = (h_x - h) // 2, (w_x - w) // 2  # x要剪裁掉的值
        hc_s, wc_s = (h_s - h) // 2, (w_s - w) // 2  # shortcut要剪裁掉的值
        x = x[..., hc_x:hc_x + h, wc_x: wc_x + w]  # center crop
        shortcut = shortcut[..., hc_s:hc_s + h, wc_s:wc_s + w]  # center crop
        return x, shortcut

    pass


class _UNetFactory(nn.Module):
    def __init__(self, encode_blocks, encode_out_channels, n_class,
                 init_encoder=True, init_decoder=True):
        """
        UNet工厂类，用于生成UNet模型的网络。
        :param encode_blocks: 列表类型。列表每个元素是一个encode的block
        :param encode_out_channels: 列表类型。列表每个元素是encode block的输出channels，按照encode的顺序。
        :param n_class: n种分类。
        :param init_encoder: 是否初始化encoder的权重。ResNet修改了encoder部分，一般不需要初始化。
        :param init_decoder: 是否初始化decoder的权重。decoder一般一样，都需要初始化。
        """
        super(_UNetFactory, self).__init__()
        self.encoder = _UNetEncoder(encode_blocks)
        self.decoder = _UNetDecoder(encode_out_channels, n_class)

        # 初始化参数
        if init_encoder:
            for m in self.encoder.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        if init_decoder:
            for m in self.decoder.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        pass

    def forward(self, x):
        x, shortcuts = self.encoder(x)
        x = self.decoder(x, shortcuts)
        return x

    pass


def unet_base(in_channels, n_class):
    """
    按照论文实现的unet网络，与论文不同的是使用了same卷积。
    :param in_channels: 输入channels，也就是image的channels
    :param n_class: n种分类
    :return: unet网络
    """
    encode_blocks = [unet_conv(in_channels, 64)]
    for i in range(4):
        encode_blocks.append(nn.Sequential(nn.MaxPool2d(2, stride=2, ceil_mode=True),
                                           unet_conv(64 * 2 ** i, 128 * 2 ** i)))
    encode_out_channels = [64, 128, 256, 512, 1024]
    return _UNetFactory(encode_blocks, encode_out_channels, n_class)


def unet_resnet(resnet_type, in_channels, n_class, pretrained=True):
    """
    用resnet预训练模型作为encoder实现的unet网络。
    :param resnet_type: resnet类型。可以是resnet18/34/50/101/152
    :param in_channels: 输入channels，也就是image的channels
    :param n_class: n种分类
    :return: 使用resnet作为backbone的unet网络
    """
    if resnet_type == 'resnet18':
        resnet = resnet18(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet34':
        resnet = resnet34(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 64, 128, 256, 512]
    elif resnet_type == 'resnet50':
        resnet = resnet50(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet101':
        resnet = resnet101(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    elif resnet_type == 'resnet152':
        resnet = resnet152(pretrained=pretrained)
        encode_out_channels = [in_channels, 64, 256, 512, 1024, 2048]
    else:
        raise ValueError('resnet type error!')
    encode_blocks = [nn.Sequential(),  # 1x，第1个encode block什么都不做
                     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # 2x，resnet的conv1_x进行第1次下采样
                     nn.Sequential(resnet.maxpool, resnet.layer1),  # 4x，resnet的maxpool进行第2次下采样，conv2_x不进行下采样
                     resnet.layer2,  # 8x，resnet的conv3_x进行第3次下采样
                     resnet.layer3,  # 16x，resnet的conv4_x进行第4次下采样
                     resnet.layer4]  # 32x，resnet的conv5_x进行第5次下采样
    return _UNetFactory(encode_blocks, encode_out_channels, n_class,
                        init_encoder=not pretrained)  # 有pretrain的encoder不初始化


if __name__ == '__main__':
    """单元测试"""
    dev = torch.device('cuda:6')
    model = unet_resnet('resnet18', 3, 8)  # resnet18作为backbone的unet
    model.to(dev)  # 装入gpu
    print(model)  # 打印看模型是否正确

    in_data = torch.randint(0, 256, (1, 3, 572, 572), dtype=torch.float)  # 测试输入
    in_data = in_data.to(dev)  # 装入gpu
    print(in_data.shape)

    out_data = model(in_data)
    print(out_data.shape)  # 输出应该是1x8x572x572的tensor
    pass
