import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from PIL import Image, ImageFilter
from utils.tools import get_proj_root


class PairCrop(object):
    def __init__(self, offsets=None, size=None):
        """
        剪裁图像
        :param offsets: 剪裁的偏移量，(H,W)类型，None表示不偏移
        :param size: 剪裁的大小，(H,W)类型，None表示不剪裁
        """
        super(PairCrop, self).__init__()
        # 偏移量可以为空或大于等于0，None或等于0表示不偏移
        assert offsets is None or (offsets[0] is None or offsets[0] >= 0) and (offsets[1] is None or offsets[1] >= 0)
        # 剪裁的大小，必须是正数或者None不剪裁
        assert size is None or (size[0] is None or size[0] > 0) and (size[1] is None or size[1] > 0)

        if offsets is None:  # HW都不偏移
            offsets = (0, 0,)
        self.start = (0 if offsets[0] is None else offsets[0],  # H或W不偏移
                      0 if offsets[1] is None else offsets[1],)

        if size is None:  # HW都不剪裁
            size = (None, None,)
        self.stop = (self.start[0] + size[0] if size[0] is not None else size[0],  # H或W不剪裁
                     self.start[1] + size[1] if size[1] is not None else size[1],)
        pass

    def __call__(self, image, label=None):
        """
        剪裁图像
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """

        image = np.asarray(image)
        if label is not None:
            label = np.asarray(label)
            assert image.shape[:2] == label.shape[:2]

        h, w = image.shape[0], image.shape[1]
        assert 0 <= self.start[0] < h and (self.stop[0] is None or 0 <= self.stop[0] < h)  # 剪裁大小不超过原图像大小
        assert 0 <= self.start[1] < w and (self.stop[1] is None or 0 <= self.stop[1] < w)

        hslice = slice(self.start[0], self.stop[0])  # H方向剪裁量
        wslice = slice(self.start[1], self.stop[1])  # W方向剪裁量

        image = Image.fromarray(image[hslice, wslice])
        if label is not None:
            label = Image.fromarray(label[hslice, wslice])
        return image, label

    pass


class PairRandomHFlip(object):
    def __init__(self):
        super(PairRandomHFlip, self).__init__()
        pass

    def __call__(self, image, label=None):
        """
        图像随机左右翻转
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """
        if random.uniform(0, 1) < 0.5:  # 50%的概率会翻转
            image = TF.hflip(image)  # 左右翻转
            if label is not None:
                label = TF.hflip(label)
        return image, label

    pass


class PairRandomVFlip(object):
    def __init__(self):
        super(PairRandomVFlip, self).__init__()
        pass

    def __call__(self, image, label=None):
        """
        图像随机上下翻转
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """
        if random.uniform(0, 1) < 0.5:  # 50%的概率会翻转
            image = TF.vflip(image)  # 上下翻转
            if label is not None:
                label = TF.vflip(label)
        return image, label

    pass


class PairAdjustColor(object):
    def __init__(self, factors=(0.3, 2.)):
        super(PairAdjustColor, self).__init__()
        self.factors = factors
        pass

    def __call__(self, image, label=None):
        """
        调整亮度、对比度、饱和度
        只调整image，不调整label
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        brightness_factor = random.uniform(*self.factors)
        contrast_factor = random.uniform(*self.factors)
        saturation_factor = random.uniform(*self.factors)

        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
        image = TF.adjust_saturation(image, saturation_factor)
        return image, label

    pass


class PairAdjustGamma(object):
    def __init__(self, gamma, gain=1):
        """
        Gamma矫正
        Out = 255*gain*(in/255)^gamma
        :param gamma: gamma 0。0~1.0
        :param gain: gain
        """
        super(PairAdjustGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain
        pass

    def __call__(self, image, label=None):
        """
        Gamma矫正
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        image = TF.adjust_gamma(image, self.gamma, self.gain)  # 只对image做gamma矫正
        return image, label

    pass


class PairResize(object):
    def __init__(self, size=None):
        """
        图像缩放
        :param size: 图像等比缩放后的大小
            如果是None，不缩放；
            如果是int，那么等比例缩放，size是缩放以后短边的长度；
            如果是tuple，那么size就是缩放后的大小(H,W)；
        """
        super(PairResize, self).__init__()
        assert size is None or isinstance(size, int) or isinstance(size, tuple)
        if isinstance(size, tuple):
            self.size = (size[1], size[0])  # 输入(H,W)，PIL要求(W,H)
        else:
            self.size = size
        pass

    def __call__(self, image, label=None):
        """
        图像等比缩放
        :param image: [H,W,C] PIL Image RGB
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB,  [H,W] PIL Image trainId
        """
        if self.size is None:  # 不缩放
            return image, label

        if isinstance(self.size, int):  # 等比例缩放
            image = TF.resize(image, self.size, interpolation=Image.BILINEAR)
            if label is not None:
                label = TF.resize(label, self.size, interpolation=Image.NEAREST)  # label要用邻近差值
        elif isinstance(self.size, tuple):  # 指定输出大小的缩放
            image = image.resize(self.size, resample=Image.BILINEAR)
            if label is not None:
                label = label.resize(self.size, resample=Image.NEAREST)  # label要用邻近差值
        return image, label

    pass


class PairNormalizeToTensor(object):
    def __init__(self, norm=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet统计的RGB mean
        IMAGE_NORM_STD = [0.229, 0.224, 0.225]  # ImageNet统计的RGB std
        LABEL_NORM_MEAN = [0.5]  # ImageNet统计的GRAY mean
        LABEL_NORM_STD = [0.5]  # ImageNet统计的GRAY std
        :param norm: 是否正则化，默认是
        :param mean: 正则化的平均值mean
        :param std: 正则化的标准差std
        """
        super(PairNormalizeToTensor, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        pass

    def __call__(self, image, label=None):
        """
        归一化，只对image除以255，label不动
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [C,H,W] tensor RGB -1.0~0.0,  [H,W] tensor trainId
        """
        # torchvision.transform的API，对PIL Image类型image归一化，也就是除以255
        # 并转为tensor，维度变为[C,H,W]
        # image [C,H,W]tensor RGB 0.0~1.0
        image = TF.to_tensor(image)

        # 正则化，x=(x-mean)/std
        # 只对image正则化, image [C,H,W]tensor RGB -1.0~1.0
        if self.norm:
            image = TF.normalize(image, self.mean, self.std)

        # 先转为ndarray，再转为tensor，不归一化，维度保持不变
        # label [H,W]tensor trainId
        if label is not None:
            label = torch.from_numpy(np.asarray(label))

        return image, label

    pass


class PairRandomFixErase(object):
    def __init__(self, mask_size=64, value=0):
        """
        按照固定大小，随机遮挡图像中的某一块方形区域
        :param mask_size: 被遮挡的区域大小，默认64x64
        :param value: 被遮挡的部分用value值填充
        """
        super(PairRandomFixErase, self).__init__()
        self.mask_size = mask_size
        self.value = value
        pass

    def __call__(self, image, label=None):
        """
        按照固定大小，随机遮挡图像中的某一块方形区域
        :param image: [C,H,W] tensor，必须是tensor
        :param label: [H,W] tensor，必须是tensor
        :return: [C,H,W] tensor,  [H,W] tensor
        """
        _, h, w = image.shape
        top = random.randint(0, h - self.mask_size)  # 随机到遮挡部分的top
        left = random.randint(0, w - self.mask_size)  # 随机到遮挡部分的left
        if random.uniform(0, 1) < 0.5:  # 随机遮挡
            image = TF.erase(image, top, left, self.mask_size, self.mask_size,
                             v=self.value, inplace=True)
        return image, label

    pass


class PairRandomGaussianBlur(object):
    def __init__(self):
        """
        随机高斯模糊
        """
        super(PairRandomGaussianBlur, self).__init__()
        pass

    def __call__(self, image, label=None):
        """
        随机高斯模糊
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [H,W,C] PIL Image RGB 0~255,  [H,W] PIL Image trainId
        """
        high = min(image.size[0], image.size[1])  # 取图像HW最小值
        radius = random.randint(1, high)  # 随机高斯模糊半径
        gaussian_filter = ImageFilter.GaussianBlur(radius=radius)  # 高斯模糊过滤器
        return image.filter(gaussian_filter), label

    pass


if __name__ == '__main__':
    x = np.array([[[53, 170, 134],
                   [92, 111, 202]],
                  [[235, 126, 244],
                   [107, 46, 15]]], dtype=np.uint8)  # 模拟一个RGB图像
    y = np.array([[2, 4],
                  [3, 5]], dtype=np.uint8)  # 模拟一个trainId的label
    print('np', x)
    print('np', y)

    x = Image.fromarray(x)
    y = Image.fromarray(y)
    x, y = PairNormalizeToTensor()(x, y)  # 测试PairNormalizeToTensor
    print('tensor', x)  # x应该是-1.0~1.0
    print('tensor', y)  # y应该是trainId
    """
    tensor tensor([[[-1.2103, -0.5424],
         [ 1.9064, -0.2856]],

        [[ 0.9405, -0.0924],
         [ 0.1702, -1.2304]],

        [[ 0.5311,  1.7163],
         [ 2.4483, -1.5430]]])
    tensor tensor([[2, 4],
    """

    res = os.path.join(get_proj_root(), 'res')
    im = Image.open(os.path.join(res, 'image.jpg'))
    lb = Image.open(os.path.join(res, 'label.png'))

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    ax = ax.flatten()
    ax[0].imshow(im)
    ax[1].imshow(lb, cmap='gray')

    crop = PairCrop(offsets=(690, None), size=(None, None))
    random_hflip = PairRandomHFlip()
    random_vflip = PairRandomVFlip()
    adjust = PairAdjustColor(factors=(0.1, 1.5))
    adjust_gamma = PairAdjustGamma(gamma=0.9)
    resize = PairResize(size=256)
    to_tensor = PairNormalizeToTensor(norm=False)
    random_fix_crop = PairRandomFixErase(mask_size=224)
    gaussian_blur = PairRandomGaussianBlur()
    ts = [
        crop,
        random_hflip,
        # random_vflip,
        # adjust,
        # adjust_gamma,
        resize,
        to_tensor,
        random_fix_crop,
        # gaussian_blur,
    ]
    for t in ts:
        im, lb = t(im, lb)
        pass

    print(im.shape, lb.shape)
    im = TF.to_pil_image(im).convert('RGB')

    ax[2].imshow(im)
    ax[3].imshow(lb, cmap='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.join(get_proj_root(), 'res'), 'aug.jpg'))
    plt.close(fig)

    pass
