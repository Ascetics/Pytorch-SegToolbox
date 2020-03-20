import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from train import get_model
from datasets.laneseg import LaneSegDataset
from datasets.process_label import gray_to_rgb
from utils.tools import now_str, get_metrics, get_confusion_matrix, get_logger
from utils.augment import PairCrop, PairNormalizeToTensor, PairResize


def test(net, test_data, device, resize_to=256, n_class=8, compare=False):
    """
    测试
    :param net: AI网络
    :param test_data: test dataset
    :param device: torch.device GPU or CPU
    :param n_class: n种分类
    :param compare: 是否生成对比图片
    :return:
    """
    net.to(device)
    net.eval()  # 测试
    confusion_matrix = np.zeros((n_class, n_class))  # 记录整个测试的混淆矩阵
    test_pics_miou = 0.  # 累加每张图像的mIoU

    offset = 690  # 剪裁690x3384
    pair_crop = PairCrop(offsets=(offset, None))  # 剪裁690x3384
    pair_resize = PairResize(size=resize_to)
    pair_norm_to_tensor = PairNormalizeToTensor(norm=True)  # 归一化并正则化

    with torch.no_grad():  # 测试阶段，不需要计算梯度，节省内存
        for i_batch, (im, lb) in enumerate(test_data):
            # if i_batch > 10:
            #     break
            im_t, lb_t = pair_crop(im, lb)  # PIL Image,PIL Image
            im_t, lb_t = pair_resize(im_t, lb_t)  # PIL Image,PIL Image
            im_t, lb_t = pair_norm_to_tensor(im_t, lb_t)  # [C,H,W]tensor,[H,W]tensor

            im_t = im_t.to(device)  # [C,H,W]tensor装入GPU
            im_t = im_t.unsqueeze(0)  # 转换为[N,C,H,W] tensor
            output = net(im_t)  # 经过模型输出[N,C,H,W] tensor
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)  # [N,H,W] tensor

            pred = pred.unsqueeze(1)  # [N,C,H,W] tensor, F.interpolate操作图像需要[N,C,H,W] tensor
            pred = pred.type(torch.float)  # 转为float数，F.interpolate只对float类型操作，int，long等都没有实现
            pred = F.interpolate(pred, size=(lb.size[1] - offset, lb.size[0]),
                                 mode='nearest')  # pred用nearest差值
            pred = pred.type(torch.uint8)  # 再转回int类型
            pred = pred.squeeze(0).squeeze(0)  # [H,W]tensor
            pred = pred.cpu().numpy()  # [H,W]ndarray

            supplement = np.zeros((offset, lb.size[0]), dtype=np.uint8)  # [H,W]ndarray,补充成背景
            pred = np.append(supplement, pred, axis=0)  # 最终的估值，[H,W]ndarray,在H方向cat，给pred补充被剪裁的690x3384
            cm = get_confusion_matrix(pred, lb, n_class)  # 本张图像的混淆矩阵
            confusion_matrix += cm  # 累加

            if compare:  # 生成对比图
                fontsize = 16  # 图像文字字体大小
                fig, ax = plt.subplots(2, 2, figsize=(20, 15))  # 画布
                ax = ax.flatten()

                ax[0].imshow(im)  # 左上角显示原图
                ax[0].set_title('Input Image', fontsize=fontsize)  # 标题

                ax[1].imshow(gray_to_rgb(np.asarray(lb)))  # 右上角显示 Grand Truth
                ax[1].set_title('Grand Truth', fontsize=fontsize)  # 标题

                pic_miou = get_metrics(cm, metrics='mean_iou')  # 计算本张图像的mIoU
                fig.suptitle('mIoU:{:.4f}'.format(pic_miou), fontsize=fontsize)  # 用mIoU作为大标题
                test_pics_miou += pic_miou

                mask = (pred != 0).astype(np.uint8) * 255  # [H,W]ndarray,alpha融合的mask

                pred = gray_to_rgb(pred)  # [H,W,C=3]ndarray RGB
                ax[3].imshow(pred)  # 右下角显示Pred
                ax[3].set_title('Pred', fontsize=fontsize)  # 标题

                mask = mask[..., np.newaxis]  # [H,W,C=1]ndarray
                pred = np.append(pred, mask, axis=2)  # [H,W,C=4]ndarray，RGB+alpha变为RGBA

                im = im.convert('RGBA')
                pred = Image.fromarray(pred).convert('RGBA')
                im_comp = Image.alpha_composite(im, pred)  # alpha融合
                ax[2].imshow(im_comp)  # 左下角显示融合图像
                ax[2].set_title('Pred over Input', fontsize=fontsize)  # 标题

                plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                                    wspace=0.01, hspace=0.01)  # 调整子图边距间距
                plt.savefig('/root/private/imfolder/pred-{:s}.jpg'.format(now_str()))  # 保存图像
                plt.close(fig)
                pass
            pass
        mean_iou = get_metrics(confusion_matrix)  # 整个测试的mIoU
        test_pics_miou /= len(test_data)

        logger = get_logger('test')
        msg = 'Test mIoU : {:.4f} (Accumulate ConfusionMat)'.format(mean_iou)
        logger.info(msg)
        msg = 'Test mIoU : {:.4f} (Mean of per Image)'.format(test_pics_miou)
        logger.info(msg)
        return mean_iou


if __name__ == '__main__':
    dev = torch.device('cuda:4')  # 选择一个可用的GPU
    load_file = ('/root/private/LaneSegmentation/weight/'
                 'deeplabv3p_xception-2020-03-18-15-32-34-epoch-03.pth')  # 读取训练好的参数
    model = get_model('deeplabv3p_xception',
                      in_channels=3, n_class=8, device=dev, load_weight=load_file)
    # model = DeepLabV3P('xception', 3, 8)
    # wt = torch.load(load_file, map_location=dev)
    # model.load_state_dict(wt)
    s = input('->')
    test(net=model,
         test_data=LaneSegDataset('/root/private/LaneSegmentation/data_list/test.csv'),  # 不剪裁，不缩放的测试集，读取PIL Image
         resize_to=384,  # 这里指定缩放大小
         n_class=8,
         device=dev,
         compare=True)
    pass
