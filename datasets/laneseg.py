import torch.utils.data as data
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from PIL import Image

from utils.augment import PairCrop, PairResize, PairAdjust, PairRandomHFlip, \
    PairNormalizeToTensor, PairRandomFixErase


class LaneSegDataset(data.Dataset):
    """
    根据输入的csv文件，读取image和label
    对label做id到trainid的转换
    并输出image和label，都是PIL Image
    """

    def __init__(self, data_list, transform=None):
        """
        :param data_list: csv文件的绝对路径，csv文件有两列，第一列是image，第二列是label
        :param transform: optional，用torchvision.transforms里的API对image和label转换
        """
        super(LaneSegDataset, self).__init__()
        self._data_frame = pd.read_csv(data_list)  # 读取传入的csv文件形成data_frame
        self._transform = transform
        pass

    def __len__(self):
        return len(self._data_frame)  # Dataset子类必须实现，返回数据集长度

    def __getitem__(self, index):
        return self._get_data(index)  # Dataset子类必须实现，通过key返回value，这里的key是索引

    def _get_data(self, index):
        image_path = self._data_frame['image'][index]  # 记录下要返回的image的路径
        label_path = self._data_frame['label'][index]  # 记录下要返回的label的路径

        image = Image.open(image_path)  # 读取image为PIL Image

        label = Image.open(label_path)  # 读取label为PIL Image
        label = np.asarray(label)  # label从PIL Image转换为ndarray
        label = id_to_trainid(label)  # label的Id转换为TrainId，这一步必须加上
        label = Image.fromarray(label.astype(np.uint8))  # label从ndarray转换为PIL Image

        if self._transform is not None:
            for t in self._transform:
                image, label = t(image, label)
                pass
        return image, label

    pass


def get_data(data_type, crop_offset=(690, None), resize_to=256,
             batch_size=1):
    """
    获取数据集
    :param data_type: 可以是'train', 'valid', 'test'数据集
    :param crop_offset: 剪裁
    :param resize_to: 等比缩放
    :param batch_size: 默认和配置文件里的BATCH_SIZE一致
    :return: 数据集DataLoader
    """
    if data_type == 'train':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairRandomHFlip(),  # 随机左右翻转
            PairAdjust(),  # 调整亮度、对比度、饱和度
            PairNormalizeToTensor(),  # 归一化正则化，变成tensor
            PairRandomFixErase(),  # 随机遮挡
        ]
        shuffle = True
    elif data_type == 'valid':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairNormalizeToTensor(),  # 归一化正则化，变成tensor
        ]
        shuffle = True
    elif data_type == 'test':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairNormalizeToTensor(norm=False),  # 归一化但不正则化，变成tensor
        ]
        shuffle = False
    else:
        raise ValueError('data type error!')

    image_dataset = LaneSegDataset(Config.DATA_LIST[data_type],
                                   transform)
    data_loader = DataLoader(image_dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=True)
    return data_loader


if __name__ == '__main__':
    # 训练、验证、测试dataset
    data = get_data('test')

    # 逐个读取，查看读取的内容，验证dataloader可用
    for i, (im, lb) in enumerate(data):
        s = input('>>>')
        if s == 'q':
            break
        print(i)
        print(type(im), im.shape)
        print(type(lb), lb.shape, np.bincount(lb.flatten()))
