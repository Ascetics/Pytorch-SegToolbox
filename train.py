import os
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from models.deeplabv3p import DeepLabV3P
from models.fcn8s import FCN8s
from datasets.laneseg import get_data
from utils.lossfn import SemanticSegLoss
from utils.tools import get_logger, timer, save_weight, get_confusion_matrix, get_metrics


@timer
def _epoch_train(net, loss_func, optimizer, data, n_class, device, i_epoch):
    """
    一个epoch训练
    :param net: AI网络
    :param loss_func: loss function
    :param optimizer: optimizer
    :param data: train data set
    :param n_class: n种分类
    :param device: torch.device CPU or GPU
    :return: loss, miou
    """
    net.to(device)
    net.train()  # 训练

    total_loss = 0.  # 一个epoch训练的loss
    total_cm = np.zeros((n_class, n_class))  # ndarray 一个epoch的混淆矩阵
    total_batch_miou = 0.

    bar_format = '{desc}{postfix}|{n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}|{elapsed}<{remaining}'
    # {desc}{进度条百分比}[{当前/总数}{用时<剩余时间}{自己指定的后面显示的}]
    tqdm_data = tqdm(data,
                     ncols=120,  # 进度条宽120列，linux必须指定，否则按照terminal宽度80
                     bar_format=bar_format,  # 进度条格式
                     desc='Epoch {:02d} Train'.format(i_epoch))  # 进度条的{desc}
    for i_batch, (im, lb) in enumerate(tqdm_data, start=1):
        im = im.to(device)  # [N,C,H,W] tensor 一个训练batch image
        lb = lb.to(device)  # [N,H,W] tensor 一个训练batch label

        optimizer.zero_grad()  # 清空梯度

        output = net(im)  # [N,C,H,W] tensor 前向传播，计算一个训练batch的output

        loss = loss_func(output, lb.type(torch.long))  # 计算一个训练batch的loss
        batch_loss = loss.detach().item()  # train过程有gradient，必须detach才能读取
        total_loss += batch_loss  # 累加训练batch的loss

        loss.backward()  # 反向传播
        optimizer.step()  # 优化器迭代

        pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N,H,W] tensor 将输出转化为dense prediction，减少一个C维度
        batch_cm = get_confusion_matrix(pred.cpu().numpy(),
                                        lb.cpu().numpy(),
                                        n_class)  # 计算混淆矩阵并累加
        total_cm += batch_cm
        batch_miou = get_metrics(batch_cm, metrics='mean_iou')
        total_batch_miou += batch_miou

        tqdm_str = 'Loss={:.4f}|mIoU={:.4f}|bat_mIoU={:.4f}'  # 进度条
        tqdm_data.set_postfix_str(
            tqdm_str.format(total_loss / i_batch,
                            get_metrics(total_cm, metrics='mean_iou'),
                            total_batch_miou / i_batch))
        pass
    total_loss /= len(data)  # float 求取一个epoch的loss
    mean_iou = get_metrics(total_cm, metrics='mean_iou')  # float 求mIoU
    total_batch_miou /= len(data)  # 计算所有batch的miou的平均

    # 记录Train日志
    log_str = ('Train Loss: {:.4f}|'
               'Train mIoU: {:.4f}|'
               'Train bat_mIoU: {:.4f}')
    log_str = log_str.format(total_loss, mean_iou, total_batch_miou)
    get_logger().info(log_str)
    return total_loss, mean_iou, total_batch_miou


@timer
def _epoch_valid(net, loss_func, data, n_class, device, i_epoch):
    """
    一个epoch验证
    :param net: AI网络
    :param loss_func: loss function
    :param data: valid data set
    :param n_class: n种分类
    :param device: torch.device CPU or GPU
    :return: loss, miou
    """
    net.to(device)
    net.eval()  # 验证

    total_loss = 0.  # 一个epoch验证的loss
    total_cm = np.zeros((n_class, n_class))  # ndarray
    total_batch_miou = 0.

    with torch.no_grad():  # 验证阶段，不需要计算梯度，节省内存
        bar_format = '{desc}{postfix}|{n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}|{elapsed}<{remaining}'
        # {desc}{进度条百分比}[{当前/总数}{用时<剩余时间}{自己指定的后面显示的}]
        tqdm_data = tqdm(data,
                         ncols=120,  # 进度条宽120列，linux必须指定，否则按照terminal宽度80
                         bar_format=bar_format,  # 进度条格式
                         desc='Epoch {:02d} Valid'.format(i_epoch))  # 进度条的{desc}
        for i_batch, (im, lb) in enumerate(tqdm_data, start=1):
            im = im.to(device)  # [N,C,H,W] tensor 一个验证batch image
            lb = lb.to(device)  # [N,H,W] tensor 一个验证batch label

            output = net(im)  # [N,C,H,W] tensor 前向传播，计算一个验证batch的output
            loss = loss_func(output, lb.type(torch.long))  # 计算一个验证batch的loss
            batch_loss = loss.detach().item()  # detach还是加上吧，
            total_loss += batch_loss  # 累加验证batch的loss

            # 验证的时候不进行反向传播
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N,H,W] tensor 将输出转化为dense prediction
            batch_cm = get_confusion_matrix(pred.cpu().numpy(),
                                            lb.cpu().numpy(),
                                            n_class)  # 计算混淆矩阵并累加
            total_cm += batch_cm
            batch_miou = get_metrics(batch_cm, metrics='mean_iou')
            total_batch_miou += batch_miou

            tqdm_str = 'Loss={:.4f}|mIoU={:.4f}|bat_mIoU={:.4f}'  # 进度条
            tqdm_data.set_postfix_str(
                tqdm_str.format(total_loss / i_batch,
                                get_metrics(total_cm, metrics='mean_iou'),
                                total_batch_miou / i_batch))
            pass
        total_loss /= len(data)  # 求取一个epoch验证的loss
        mean_iou = get_metrics(total_cm, metrics='mean_iou')  # float 求mIoU
        total_batch_miou /= len(data)

        # 记录Valid日志
        log_str = ('Valid Loss: {:.4f}|'
                   'Valid mIoU: {:.4f}|'
                   'Valid bat_mIoU: {:.4f}')
        log_str = log_str.format(total_loss, mean_iou, total_batch_miou)
        get_logger().info(log_str)
        return total_loss, mean_iou, total_batch_miou


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
        get_logger().info('Epoch: {:02d}'.format(e))
        # 一个epoch训练
        _epoch_train(net, loss_func, optimizer, train_data, n_class, device, e)
        # 每个epoch的参数都保存
        save_dir = save_weight(net, model_name, e)
        get_logger().info(save_dir)  # 日志记录
        # 一个epoch验证
        _epoch_valid(net, loss_func, valid_data, n_class, device, e)
        pass
    pass


def get_model(model_type, in_channels, n_class, device, load_weight=None):
    """
    获取AI网络
    :param model_type: 网络类型
    :param in_channels: 输入图像通道数
    :param n_class: n种分类
    :param device: torch.device GPU or CPU
    :param load_weight: string已有权重文件的绝对路径，有就加载，默认没有
    :return:
    """
    if model_type == 'fcn8s':
        # raise NotImplementedError
        model = FCN8s(n_class)
    elif model_type == 'unet_resnet152':
        raise NotImplementedError
        # model = unet_resnet('resnet152', in_channels, n_class, pretrained=True)
    elif model_type == 'deeplabv3p_resnet50':
        model = DeepLabV3P('resnet50', in_channels, n_class)
    elif model_type == 'deeplabv3p_resnet101':
        model = DeepLabV3P('resnet101', in_channels, n_class)
    elif model_type == 'deeplabv3p_xception':
        model = DeepLabV3P('xception', in_channels, n_class)
    else:
        raise ValueError('model name error!')
    get_logger().info('-' * 32 + str(model_type) + '-' * 32)

    model.to(device)

    if load_weight is None:
        get_logger().info('Load weight is not specified!')
    elif os.path.exists(load_weight):
        # 有训练好的模型就加载
        get_logger().info(load_weight + ' exists! loading...')
        wt = torch.load(load_weight, map_location=device)
        model.load_state_dict(wt)
    else:
        get_logger().info(load_weight + ' can not be found!')
        pass
    return model


if __name__ == '__main__':
    dev = torch.device('cuda:0')
    # name = 'deeplabv3p_resnet101'
    name = 'deeplabv3p_xception'
    # name = 'fcn8s'
    # load_file = None
    load_file = ('/home/mist/Pytorch-SegToolbox/res/'
                 'deeplabv3p_xception-2020-03-26-16-47-28-epoch-01.pth')

    num_class = 8
    mod = get_model(name, 3, num_class, dev, load_file)
    mod.to(dev)

    lossfn = SemanticSegLoss('cross_entropy+dice', dev)
    lossfn.to(dev)

    optm = torch.optim.Adam(params=mod.parameters(),
                            lr=0.003)  # 将模型参数装入优化器

    # 768x256,1024x384,1536x512
    # 1020x3384,510x1692,255x846
    train(net=mod,
          loss_func=lossfn,
          optimizer=optm,
          train_data=get_data('train', resize_to=578, batch_size=4),
          valid_data=get_data('valid', resize_to=578, batch_size=4),
          n_class=num_class,
          device=dev,
          model_name=name,
          epochs=40)  # 开始训（炼）练（丹）
