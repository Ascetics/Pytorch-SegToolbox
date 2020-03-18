import os
import numpy as np
import torch
import torch.nn.functional as F

from models.deeplabv3p import DeepLabV3P
from datasets.laneseg import get_data
from utils.lossfn import SemanticSegLoss
from utils.tools import log, now_str, epoch_timer, save_weight, get_confusion_matrix, get_metrics
from config import Config


@epoch_timer
def _epoch_train(net, loss_func, optimizer, train_data, n_class, device):
    """
    一个epoch训练
    :param net: AI网络
    :param loss_func: loss function
    :param optimizer: optimizer
    :param train_data: train data set
    :param n_class: n种分类
    :param device: torch.device CPU or GPU
    :return: loss, miou
    """
    net.to(device)
    net.train()  # 训练

    total_loss = 0.  # 一个epoch训练的loss
    confusion_matrix = np.zeros((n_class, n_class))  # ndarray 一个epoch的混淆矩阵

    for i_batch, (im, lb) in enumerate(train_data):
        im = im.to(device)  # [N,C,H,W] tensor 一个训练batch image
        lb = lb.to(device)  # [N,H,W] tensor 一个训练batch label

        optimizer.zero_grad()  # 清空梯度
        output = net(im)  # [N,C,H,W] tensor 前向传播，计算一个训练batch的output
        loss = loss_func(output, lb.type(torch.int64))  # 计算一个训练batch的loss
        total_loss += loss.detach().item()  # train过程有gradient，必须detach才能读取，累加训练batch的loss
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器迭代
        # _adjust_lr(epoch, i, len(self.train_data))  # 优化器迭代后调整学习率

        pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N,H,W] tensor 将输出转化为dense prediction，减少一个C维度
        confusion_matrix += get_confusion_matrix(pred.cpu().numpy(),
                                                 lb.cpu().numpy(),
                                                 n_class)  # 计算混淆矩阵并累加
        del im, lb, pred  # 节省内存
        pass
    total_loss /= len(train_data)  # float 求取一个epoch的loss
    mean_iou = get_metrics(confusion_matrix, metrics='mean_iou')  # float 求mIoU

    return total_loss, mean_iou


@epoch_timer
def _epoch_valid(net, loss_func, valid_data, n_class, device):
    """
    一个epoch验证
    :param net: AI网络
    :param loss_func: loss function
    :param valid_data: valid data set
    :param n_class: n种分类
    :param device: torch.device CPU or GPU
    :return: loss, miou
    """
    net.to(device)
    net.eval()  # 验证

    total_loss = 0.  # 一个epoch验证的loss
    confusion_matrix = np.zeros((n_class, n_class))  # ndarray

    with torch.no_grad():  # 验证阶段，不需要计算梯度，节省内存
        for i_batch, (im, lb) in enumerate(valid_data):
            im = im.to(device)  # [N,C,H,W] tensor 一个验证batch image
            lb = lb.to(device)  # [N,H,W] tensor 一个验证batch label

            output = net(im)  # [N,C,H,W] tensor 前向传播，计算一个验证batch的output
            loss = loss_func(output, lb.type(torch.int64))  # 计算一个验证batch的loss
            total_loss += loss.detach().item()  # detach还是加上吧，累加验证batch的loss

            # 验证的时候不进行反向传播
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N,H,W] tensor 将输出转化为dense prediction
            confusion_matrix += get_confusion_matrix(pred.cpu().numpy(),
                                                     lb.cpu().numpy(),
                                                     n_class)  # 计算混淆矩阵并累加
            del im, lb, pred  # 节省内存
            pass
        total_loss /= len(valid_data)  # 求取一个epoch验证的loss
        mean_iou = get_metrics(confusion_matrix, metrics='mean_iou')  # float 求mIoU
        return total_loss, mean_iou


def train(net, loss_func, optimizer, train_data, valid_data,
          n_class, device, model_name, epochs=20):
    """
    训练
    :param net: AI网络
    :param loss_func: loss function
    :param optimizer: optimizer
    :param train_data: train data set
    :param valid_data: valid data set
    :param n_class: n种分类
    :param device: torch.device CPU or GPU
    :param model_name: 用于保存模型权重
    :param epochs: 训练多少个EPOCH
    :return:
    """
    for e in range(1, epochs + 1):
        epoch_str = '{:s}|Epoch: {:02d}|\n'.format(str(now_str()), e)
        log(epoch_str)

        # 一个epoch训练
        t_loss, t_miou = _epoch_train(net, loss_func, optimizer, train_data, n_class, device)
        train_str = 'Train Loss: {:.4f}|Train mIoU: {:.4f}|\n'.format(t_loss, t_miou)
        log(train_str)

        # 每个epoch的参数都保存
        save_dir = save_weight(net, model_name, e)
        log(save_dir + '\n')  # 日志记录

        # 一个epoch验证
        v_loss, v_miou = _epoch_valid(net, loss_func, valid_data, n_class, device)
        valid_str = 'Valid Loss: {:.4f}|Valid mIoU: {:.4f}|\n'.format(v_loss, v_miou)
        log(valid_str)
        pass
    pass


def get_model(model_type, in_channels, n_class, device, load_weight=None):
    """
    获取AI网络
    :param model_type: 网络类型
    :param in_channels: 输入图像通道数
    :param n_class: n种分类
    :param load_weight: string已有权重文件的绝对路径，有就加载，默认没有
    :param remake_data: 重新生成data list，默认不生成
    :return:
    """
    if model_type == 'fcn8s':
        raise NotImplementedError
        # model = FCN8s(n_class)
    elif model_type == 'resnet152':
        raise NotImplementedError
        # model = unet_resnet('resnet152', in_channels, n_class, pretrained=True)
    elif model_type == 'deeplabv3p_resnet':
        raise NotImplementedError
        # model = DeepLabV3P('resnet101', in_channels, n_class)
    elif model_type == 'deeplabv3p_xception':
        model = DeepLabV3P('xception', in_channels, n_class)
    else:
        raise ValueError('model name error!')
    log(model_type + '\n')

    model.to(device)

    if load_weight and os.path.exists(load_weight):
        # 有训练好的模型就加载
        log(load_weight + ' exists! loading...\n')
        wt = torch.load(load_weight, map_location=device)
        model.load_state_dict(wt)
    else:
        print(load_weight + ' can not be found!')

    return model


if __name__ == '__main__':
    # name = 'deeplabv3p_resnet'
    # load_file = None
    # load_file = '/root/private/LaneSegmentation/weight/deeplabv3p_resnet-2020-03-10 15:09:24.382447-epoch-01.pkl'

    # name = 'fcn8s'
    # load_file = None

    name = 'deeplabv3p_xception'
    # load_file = None
    load_file = ('/root/private/LaneSegmentation/weight/'
                 'deeplabv3p_xception-2020-03-17 06:03:02.609908-epoch-14.pth')

    num_class = 8
    custom_model = get_model(name, 3, num_class, Config.DEVICE, load_file)
    custom_model.to(Config.DEVICE)

    custom_loss_func = SemanticSegLoss('cross_entropy+dice', Config.DEVICE)
    custom_loss_func.to(Config.DEVICE)

    custom_optimizer = torch.optim.Adam(params=custom_model.parameters(),
                                        lr=Config.LR)  # 将模型参数装入优化器

    # 768x256,1024x384,1536x512
    train(net=custom_model,
          loss_func=custom_loss_func,
          optimizer=custom_optimizer,
          train_data=get_data('train', resize_to=384, batch_size=Config.TRAIN_BATCH_SIZE),
          valid_data=get_data('valid', resize_to=384, batch_size=Config.TRAIN_BATCH_SIZE),
          n_class=num_class,
          device=Config.DEVICE,
          model_name=name,
          epochs=20)  # 开始训（炼）练（丹）
