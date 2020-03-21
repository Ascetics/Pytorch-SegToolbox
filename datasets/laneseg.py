import pandas as pd
import numpy as np
import os
import sklearn

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.tools import get_proj_root
from utils.augment import PairCrop, PairResize, PairAdjustColor, PairRandomHFlip, \
    PairNormalizeToTensor, PairRandomFixErase
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


class LaneSegDataset(Dataset):
    """
    根据输入的csv文件，读取image和label
    对label做id到trainid的转换
    并输出image和label，都是PIL Image
    """
    image_file_base = '/root/data/LaneSeg/Image_Data'  # image文件的根目录
    label_file_base = '/root/data/LaneSeg/Gray_Label'  # label文件的根目录

    _data_list_dir = os.path.join(
        os.path.join(get_proj_root(), 'datasets'), 'data_list')
    _cvs_files = {  # csv文件的绝对路径，csv文件有两列，第一列是image，第二列是label
        'train': os.path.join(_data_list_dir, 'laneseg_train.csv'),  # 车道线分割训练集csv文件路径
        'valid': os.path.join(_data_list_dir, 'laneseg_valid.csv'),  # 车道线分割验证集csv文件路径
        'test': os.path.join(_data_list_dir, 'laneseg_test.csv'),  # 车道线分割测试集csv文件路径
    }

    @staticmethod
    def _get_image_label_dir():
        """
        遍历服务器image和label目录，将image和label一一对应
        :return: 生成器（image绝对路径，label绝对路径）
        """
        data_err = 'data error. check!'
        image_base = os.path.join(LaneSegDataset.image_file_base)  # 服务器上Image根目录
        label_base = os.path.join(LaneSegDataset.label_file_base)  # 服务器上Label根目录

        for road in os.listdir(image_base):  # 遍历根目录下所有目录
            image_road = os.path.join(image_base, road)  # image的Road02-Road04
            label_road = os.path.join(label_base, 'Label_' + str.lower(road))  # label的Label_road02-Label_road04
            if not (os.path.isdir(image_road) and
                    os.path.exists(label_road) and
                    os.path.isdir(label_road)):
                print(image_road, label_road, data_err)  # 路径不存在打印显示，跳过
                continue
            for record in os.listdir(image_road):  # 遍历road下所有目录
                image_record = os.path.join(image_road, record)  # image的 Record001-Record007
                label_record = os.path.join(label_road, 'Label/' + record)  # label的Record001-Record007，比image多了一层Label
                if not (os.path.isdir(image_record) and
                        os.path.exists(label_record) and
                        os.path.isdir(label_record)):
                    print(image_record, label_record, data_err)  # 路径不存在打印显示，跳过
                    continue
                for camera in os.listdir(image_record):  # 遍历record下所有目录
                    image_camera = os.path.join(image_record, camera)  # image的Camera5-Camera6
                    label_camera = os.path.join(label_record, camera)  # label的Camera5-Camera6
                    if not (os.path.isdir(image_camera) and
                            os.path.exists(label_camera) and
                            os.path.isdir(label_camera)):
                        print(image_camera, label_camera, data_err)  # 路径不存在打印显示，跳过
                        continue
                    for image in os.listdir(image_camera):  # 遍历Camera下所有图片
                        image_abspath = os.path.join(image_camera, image)  # image
                        label_abspath = os.path.join(label_camera,
                                                     image.replace('.jpg', '_bin.png'))  # label名字比image多_bin，格式png
                        if not (os.path.isfile(image_abspath) and
                                os.path.exists(label_abspath) and
                                os.path.isfile(label_abspath)):
                            print(image_abspath, label_abspath, data_err)  # 图片不存在或不对应打印显示，跳过
                            continue
                        yield image_abspath, label_abspath  # 生成器函数返回
        pass

    @staticmethod
    def make_data_list(train_rate=0.7, valid_rate=0.2, shuffle=True):
        """
        打乱顺序，生成data_list的csv文件。
        :param train_path: 训练集保存路径
        :param valid_path: 验证集保存路径
        :param test_path: 测试集保存路径
        :param train_rate: 训练集占比，默认0.7
        :param valid_rate: 验证集占比，默认0.2
        :return:
        """
        g = LaneSegDataset._get_image_label_dir()  # 获取生成器
        abspaths = list(g)  # 将生成器转换为列表

        df = pd.DataFrame(
            data=abspaths,  # csv文件数据，每个元素是一条数据
            columns=['image', 'label']  # 两列 image、label
        )
        if shuffle:
            df = sklearn.utils.shuffle(df)  # 随机打乱顺序
        train_size = int(df.shape[0] * train_rate)
        valid_size = int(df.shape[0] * valid_rate)

        print('total: {:d} | train: {:d} | val: {:d} | test: {:d}'.format(
            df.shape[0], train_size, valid_size,
            df.shape[0] - train_size - valid_size))

        df_train = df[0: train_size]  # train数据集
        df_valid = df[train_size: train_size + valid_size]  # valid数据集
        df_test = df[train_size + valid_size:]  # test数据集

        df_train.to_csv(os.path.join(LaneSegDataset._cvs_files['train']), index=False)  # 保存train.csv文件
        df_valid.to_csv(os.path.join(LaneSegDataset._cvs_files['valid']), index=False)  # 保存valid.csv文件
        df_test.to_csv(os.path.join(LaneSegDataset._cvs_files['test']), index=False)  # 保存test.csv文件
        pass

    def __init__(self, dataset_type, transform=None):
        """
        :param dataset_type: 数据集类型，可以是'train', 'valid', 'test'
        :param transform: optional，用augment里的API对image和label转换
        """
        super(LaneSegDataset, self).__init__()
        assert dataset_type in ('train', 'valid', 'test')

        self._dataset_type = dataset_type
        self._data_frame = pd.read_csv(
            LaneSegDataset._cvs_files[self._dataset_type])  # 读取传入的csv文件形成data_frame
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
        label = self.encode(label)  # label的Id转换为TrainId，这一步必须加上
        label = Image.fromarray(label.astype(np.uint8))  # label从ndarray转换为PIL Image

        if self._transform is not None:
            for t in self._transform:
                image, label = t(image, label)
                pass
        return image, label

    @staticmethod
    def encode(label):
        """
        要求输入的图像是灰度图，灰阶0-255。
        matplotlib.img读取灰度图像的灰阶是0.0-1.0，需要自己处理到0-255
        :param label: [H,W,C] ndarray 或者 PIL Image，id标注的label图像
        :return: trainId标注的label图像
        """
        label = np.asarray(label)
        encoded_label = np.zeros(label.shape, dtype=np.uint8)

        # trainId = 0
        # dst[src == 0] = 0
        encoded_label[label == 255] = 0
        encoded_label[label == 249] = 0

        # trainId = 1
        encoded_label[label == 200] = 1
        encoded_label[label == 204] = 1
        encoded_label[label == 213] = 0  # ignoreInEval
        encoded_label[label == 209] = 1
        encoded_label[label == 206] = 0
        encoded_label[label == 207] = 0

        # trainId = 2
        encoded_label[label == 201] = 2
        encoded_label[label == 203] = 2
        encoded_label[label == 211] = 0
        encoded_label[label == 208] = 0

        # trainId = 3
        encoded_label[label == 216] = 0
        encoded_label[label == 217] = 3
        encoded_label[label == 215] = 0

        # trainId = 4
        encoded_label[label == 218] = 0
        encoded_label[label == 219] = 0

        # trainId = 5->4,因trainId=4都被忽略，5递进为4，后面一样递进
        encoded_label[label == 210] = 4
        encoded_label[label == 232] = 0

        # trainId = 6->5
        encoded_label[label == 214] = 5

        # trainId = 7->6
        encoded_label[label == 202] = 0
        encoded_label[label == 220] = 6
        encoded_label[label == 221] = 6
        encoded_label[label == 222] = 6
        encoded_label[label == 231] = 0
        encoded_label[label == 224] = 6
        encoded_label[label == 225] = 6
        encoded_label[label == 226] = 6
        encoded_label[label == 230] = 0
        encoded_label[label == 228] = 0
        encoded_label[label == 229] = 0
        encoded_label[label == 233] = 0

        # trainId = 8->7
        encoded_label[label == 205] = 7
        encoded_label[label == 212] = 0
        encoded_label[label == 227] = 7
        encoded_label[label == 223] = 0
        encoded_label[label == 250] = 7

        return encoded_label

    @staticmethod
    def decode(label):
        """
        要求输入的图像是灰度读图，灰阶0-255。
        matplotlib.img读取灰度图像的灰阶是0.0-1.0，需要自己处理到0-255
        :param label: trainId标注的label图像
        :return: id标注的label图像
        """
        label = np.asarray(label)
        decoded_label = np.zeros(label.shape, dtype=np.uint8)

        decoded_label[label == 0] = 0
        decoded_label[label == 1] = 200
        decoded_label[label == 2] = 201
        decoded_label[label == 3] = 217
        # id = 4->5,因trainId=4都被忽略，5递进为4，转换为id需要处理，后面一样递进
        decoded_label[label == 4] = 210
        decoded_label[label == 5] = 214
        decoded_label[label == 6] = 228
        decoded_label[label == 7] = 205

        return decoded_label

    @staticmethod
    def decode_rgb(label_gray):
        """
        将trainId标注的灰度label转换成rgb彩色label
        :param label_gray: trainId标注灰度图label
        :return: rgb彩色label
        """
        label_gray = np.asarray(label_gray)
        height, width = label_gray.shape
        rgb = np.zeros((height, width, 3), dtype=np.uint8)

        rgb[label_gray == 0] = np.array([0, 0, 0])
        rgb[label_gray == 1] = np.array([70, 130, 180])  # 浅蓝色
        rgb[label_gray == 2] = np.array([0, 0, 142])  # 深蓝色
        rgb[label_gray == 3] = np.array([220, 220, 0])  # 黄色
        # id = 4->5,因trainId=4都被忽略，5递进为4，转换为RGBlabel的时候要处理，后面一样递进
        rgb[label_gray == 4] = np.array([128, 64, 128])  # 紫色
        rgb[label_gray == 5] = np.array([190, 153, 153])  # 浅粉色人行道
        rgb[label_gray == 6] = np.array([51, 255, 51])  # 绿色
        rgb[label_gray == 7] = np.array([255, 128, 0])  # 橘黄色

        return rgb

    pass


def get_data(dataset_type, crop_offset=(690, None), resize_to=256,
             batch_size=1):
    """
    获取数据集
    :param dataset_type: 可以是'train', 'valid', 'test'数据集
    :param crop_offset: 剪裁
    :param resize_to: 等比缩放
    :param batch_size: 默认和配置文件里的BATCH_SIZE一致
    :return: 数据集DataLoader
    """
    if dataset_type == 'train':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairRandomHFlip(),  # 随机左右翻转
            PairAdjustColor(),  # 调整亮度、对比度、饱和度
            PairNormalizeToTensor(),  # 归一化正则化，变成tensor
            PairRandomFixErase(),  # 随机遮挡
        ]
        shuffle = True
    elif dataset_type == 'valid':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairNormalizeToTensor(),  # 归一化正则化，变成tensor
        ]
        shuffle = True
    elif dataset_type == 'test':
        transform = [
            PairCrop(offsets=crop_offset),  # 剪裁
            PairResize(size=resize_to),  # 等比缩放
            PairNormalizeToTensor(norm=False),  # 归一化但不正则化，变成tensor
        ]
        shuffle = False
    else:
        raise ValueError('data type error!')

    image_dataset = LaneSegDataset(dataset_type, transform)
    data_loader = DataLoader(image_dataset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=True)
    return data_loader


if __name__ == '__main__':
    # LaneSegDataset.make_data_list()

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

        fig, ax = plt.subplots(1, 2)
        ax = ax.flatten()
        im = TF.to_pil_image(im.squeeze(0))
        lb = TF.to_pil_image(lb.squeeze(0))
        ax[0].imshow(im)
        ax[1].imshow(lb, cmap='gray')
        plt.savefig(os.path.join(os.path.join(get_proj_root(), 'res'), 'laneseg_dataset.jpg'))
        plt.close(fig)


