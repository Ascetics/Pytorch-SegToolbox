import torch
import torch.nn as nn


class FCN8s(nn.Module):
    """
    FCN8s模型，backbone为VGG16，代码全部是自己手写的
    From mine own
    https://github.com/Ascetics/LaneSegmentation/blob/master/nets/fcn8s.py
    https://github.com/Ascetics/LaneSegmentation/blob/master/nets/fcn8s.txt
    """

    def __init__(self, n_class):
        """
        FCN8s的__init__
        :param n_class: 分类个数
        """
        super(FCN8s, self).__init__()
        # conv1 第一次下采样，padding=100保证fc6的7x7卷积能正常进行
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        )

        # conv2 第二次下采样
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        )

        # conv3 第三次下采样
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        )

        # conv4 第四次下采样
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        )

        # conv5 第五次下采样
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        )

        # fc6 将全连接层改为卷积层，7x7卷积
        # 当输入大于224x224时，输出heatmap
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc8
        self.fc8 = nn.Conv2d(4096, n_class, 1)

        # 上采样2倍，与conv4融合
        self.upsample2_conv4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                                  bias=False)
        self.adjust4 = nn.Conv2d(512, n_class, 1)

        # 上采样2倍，与conv3融合
        self.upsample2_conv3 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                                  bias=False)
        self.adjust3 = nn.Conv2d(256, n_class, 1)

        # 上采样8倍
        self.upsample8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                            bias=False)
        self._init_param()

    def _init_param(self):
        """
        初始化参数
        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        pass

    def forward(self, x):
        # 保留输入，用于最后剪裁
        h = x

        # 下采样
        h = self.conv1(h)  # 1/2
        h = self.conv2(h)  # 1/4
        h = self.conv3(h)  # 1/8
        conv3 = h  # 保留conv3，用于第二次特征融合
        h = self.conv4(h)  # 1/16
        conv4 = h  # 保留conv4，用于第一次特征融合
        h = self.conv5(h)  # 1/32
        h = self.fc6(h)
        h = self.fc7(h)
        h = self.fc8(h)

        # 上采样
        h = self.upsample2_conv4(h)  # 上采样2倍
        adjust4 = self.adjust4(conv4)  # 调整conv4的channel，与2倍上采样channel一致
        adjust4 = adjust4[..., 5:5 + h.shape[2], 5:5 + h.shape[3]]  # 剪裁
        h = h + adjust4  # 第一次与conv4（16s）特征融合

        h = self.upsample2_conv3(h)  # 特征融合后上采样2倍
        adjust3 = self.adjust3(conv3)  # 调整conv3的channel，与融合后2倍上采样channel一致
        adjust3 = adjust3[..., 9:9 + h.shape[2], 9:9 + h.shape[3]]  # 剪裁
        h = h + adjust3  # 第二次与conv3（8s）特征融合

        h = self.upsample8(h)  # 融合后上采样8倍
        h = h[..., 31:31 + x.shape[2], 31:31 + x.shape[3]].contiguous()  # 剪裁
        return h

    pass


if __name__ == '__main__':
    # 不知道怎么训练，不知道怎么初始化权重，不知道loss的含义
    fcn8s = FCN8s(8)
    in_data = torch.randint(0, 10, (1, 3, 224, 224)).type(torch.float)
    out_data = fcn8s(in_data)
