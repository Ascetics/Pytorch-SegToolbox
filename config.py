import torch


class Config(object):
    """
    配置类
    """
    # 设备   ####################################################################
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')
    DEVICE = torch.device('cuda:6')

    # 超参数 ####################################################################
    TRAIN_BATCH_SIZE = 4  # batch大小
    LR = 0.003  # 学习率
    LR_MIN = 1e-6  # 最小学习率
    WEIGHT_DECAY = 0.0001
    EPOCHS = 50  # 训练次数
    # 数据集 ####################################################################
    DATASETS_ROOT = '/root/private/torch_datasets'  # Pytorch数据集根目录

    # 训练结果###################################################################
    WEIGHT_SAVE_PATH = '/root/private/LaneSegmentation/weight'  # weight保存路径
    LOG_FILE = '/root/private/LaneSegmentation/weight/train.log'  # 训练日志

    # 数据处理 ####################################################################
    IMAGE_BASE = '/root/data/LaneSeg/Image_Data'  # image文件的根目录
    LABEL_BASE = '/root/data/LaneSeg/Gray_Label'  # label文件的根目录

    TRAIN_RATE = 0.7  # 数据集划分，训练集占整个数据集的比例
    VALID_RATE = 0.2  # 数据集划分，验证集占整个数据集的比例

    pass
