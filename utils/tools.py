import os
import random
import time
import torch
import numpy as np
from datetime import datetime
from config import Config


def now_str():
    """
    返回格式化的当前日期时间字符串
    :return: YYYY-mm-dd-HH-MM-SS
    """
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


def epoch_timer(func):
    """
    装饰器。epoch计时器，记录一个epoch用时并打印
    :param func: 被装饰函数，是epoch_train
    :return:
    """

    def timer(*args, **kwargs):  # func的所有入参
        begin_time = datetime.now()  # 开始时间
        res = func(*args, **kwargs)  # 执行func，记录func返回值
        end_time = datetime.now()  # 结束时间
        mm, ss = divmod((end_time - begin_time).seconds, 60)  # 秒换算成分、秒
        hh, mm = divmod(mm, 60)  # 分钟换算成时、分
        duration_str = 'Duration: {:02d}:{:02d}:{:02d}|'.format(hh, mm, ss)  # HH:mm:ss
        log(duration_str)  # 记录到日志里面
        return res  # 返回func返回值

    return timer


def log(msg, logfile=Config.LOG_FILE):
    """
    日志记录
    :param msg: 要记录的内容
    :param logfile: 日志文件，如果没有就创建
    :return:
    """
    log_f = open(logfile, mode='a')
    log_f.write(msg)  # 写到日志
    log_f.flush()  # 很有必要的马上写
    log_f.close()
    print(msg, end='', flush=True)  # 马上打印到终端
    pass


def save_weight(net, name, epoch, save_dir=Config.WEIGHT_SAVE_PATH):
    """
    保存模型参数，模型参数文件名格式 {模型名}-{保存日期时间}-epoch-{第几个epoch}.pkl
    :param net: 要保存参数的模型
    :param name: 模型的名字
    :param epoch: 训练到第几个epoch的参数
    :param save_dir: 保存模型参数的路径，不包含文件名
    :return:
    """
    filename = '{:s}-{:s}-epoch-{:02d}.pth'
    filename = filename.format(name, now_str(), epoch)  # 模型参数文件{模型名}-{保存日期时间}-epoch-{第几个epoch}.pkl
    save_dir = os.path.join(save_dir, filename)  # 保存的文件绝对路径
    torch.save(net.state_dict(), save_dir)  # 保存模型参数
    return save_dir


def get_confusion_matrix(pred, label, n_class=8):
    """
    计算图像的混淆矩阵
    cm = np.bincount(n_class * label[mask] + pred[mask], minlength=n_class ** 2)
    n_class * label[mask] 将0,1,2,...,n映射到0,1,2,..,n*n
    n_class * label[mask] + pred[mask] 将分类映射到不同的grand true
    比如
    n=8
    label=0,pred=0,1,2,...,7映射后是0,1,2,...,7
    label=1,pred=0,1,2,...,7映射后是8,9,10,...,15
    ...
    label=7,pred=0,1,2,...,7映射后是56,57,58,...,63
    bincount 统计以后就会对应到0,1,2,...,63，再reshape就得到了混淆矩阵
    :param pred: 估值 维度NHW
    :param label: grand true 维度NHW
    :param n_class: n种分类
    :return: 混淆矩阵
        [n_class, n_class]ndarray混淆矩阵
        dim=0是grand truth
        dim=1是predict
    """
    pred = np.asarray(pred).astype(np.long)  # 必须转化为int或long类型否则bincount报错
    label = np.asarray(label).astype(np.long)
    assert pred.shape == label.shape

    mask = (label >= 0) & (label < n_class)  # 转化为一维向量
    cm = np.bincount(label[mask] * n_class + pred[mask], minlength=n_class ** 2)  # 计算混淆矩阵
    cm = cm.reshape((n_class, n_class))  # reshape成矩阵的confusion matrix

    return cm  #


def get_metrics(cm, metrics='mean_iou'):
    """
    计算评价指标
    :param cm: [n_class, n_class]ndarray混淆矩阵
        dim=0是grand truth
        dim=1是predict
    :param metrics: 评价指标，可以取值的有
        'accuracy': 准确率
                    accuracy=(TP+TN)/(TP+TN+FP+FN) 所有判别(TP+TN+FP+FN)中，判别正确(TP+TN)占的比例。
        'mean_iou': 平均交并比 mean_iou是所有类别iou的均值
                    iou=(TP)/(TP+FP+FN) 所有判定为正类(TP+FP)和所有正类(TP+FN)，交集占并集的比例。
                    常用于目标检测、语义分割
        'recall': 召回率
                  recall=(TP)/(TP+FN) 所有正类(TP+FN)中，真实的正类(TP)占的比例。
        'precision': 精确率
                     precision=(TP)/(TP+FP) 所有被判定为正类(TP+FP)中，真实的正类(TP)占的比例。
    :return: 相应的metrics
    """
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1]

    if metrics == 'mean_iou':
        old_settings = np.seterr(divide='ignore', invalid='ignore')  # 忽略除零或无效的浮点数操作，记录旧的设置
        iou = np.diag(cm) / (np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm))  # 每个分类的交并比
        mean_iou = np.nanmean(iou)
        np.seterr(**old_settings)  # 恢复旧的设置
        return mean_iou  # float平均交并比
    elif metrics == 'accuracy':
        acc = np.sum(np.diag(cm), axis=0) / np.sum(np.sum(cm, axis=0), axis=0)
        if np.isnan(acc):  # 正确率不应出现除0情况，使用nan默认的warning
            acc = 0.
        return acc  # float准确率
    elif metrics == 'recall':
        recall = np.diag(cm) / np.sum(cm, axis=1)
        return recall  # n_class ndarray
    elif metrics == 'precision':
        precision = np.diag(cm) / np.sum(cm, axis=0)
        return precision  # n_class ndarry
    else:
        raise ValueError('metrics type error!')
    pass


if __name__ == '__main__':
    p = np.array([[[[1, 2],
                    [2, 0]]]])
    p = torch.from_numpy(p)

    lb = np.array([[[[0, 1],
                     [2, 0]]]])
    lb = torch.from_numpy(lb)
    c = get_confusion_matrix(p, lb, 3)
    print(c)
    print('accuracy', get_metrics(c, metrics='accuracy'))
    print('mean_iou', get_metrics(c, metrics='mean_iou'))
    print('recall', get_metrics(c, metrics='recall'))
    print('precision', get_metrics(c, metrics='precision'))


    @epoch_timer
    def test_train():  # 模拟一个epoch训练
        time.sleep(random.randint(0, 5) / 5)
        return random.random(), random.random()


    @epoch_timer
    def test_valid():  # 模拟一个epoch验证
        time.sleep(random.randint(0, 5) / 5)
        return random.random(), random.random()


    epochs = 5
    for e in range(1, epochs + 1):
        epoch_str = '{:s}|Epoch: {:02d}|'.format(str(datetime.now()), e)
        log(epoch_str, './test.log')

        # 模拟一个epoch训练
        t_loss, t_miou = test_train()
        train_str = 'Train Loss: {:.4f}|Train mIoU: {:.4f}|'.format(t_loss, t_miou)
        log(train_str, './test.log')

        # 模拟一个epoch验证
        v_loss, v_miou = test_valid()
        valid_str = 'Valid Loss: {:.4f}|Valid mIoU: {:.4f}|'.format(v_loss, v_miou)
        log(valid_str, './test.log')
        log('\n', './test.log')
        pass
    pass
