import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datasets.laneseg import LaneSegDataset
from PIL import Image


def observe_data(image, label, name=None):
    """
    将image和label融合到一张图片保存
    :param image: PIL Image，image
    :param label: PIL Image，label
    :param alpha: deprecated
        曾经想用PIL.Image.blend；
        后来发现alpha_composite更好就废弃了；
    :param name: 融合图像的文件名，没有就按照默认的方式命名
    :return:
    """
    image = np.array(image)  # PIL Image -> [H,W,C]ndarray
    label = np.asarray(label)  # PIL Image -> [H,W]ndarray id
    label = LaneSegDataset.encode(label)  # [H,W]ndarray trainId
    mask = (label != 0).astype(np.uint8) * 255  # [H,W]透明部分alpha的mask

    h, w = label.shape[:2]

    wmax = np.max(label, axis=1)  # 找到每行最大元素，有特征的行>0
    wmax = wmax != 0  # 判断每行有无特征
    top = wmax.argmax(axis=0)  # 找到第一个有特征的行号
    bottom = h - np.flip(wmax, axis=0).argmax(axis=0) - 1  # 找到最后一个有特征的行号

    hmax = np.max(label, axis=0)  # 找到每列最大元素，有特征>0
    hmax = hmax != 0  # 判断每列有无特征
    left = hmax.argmax(axis=0)  # 找到第一个有特征的列号
    right = w - np.flip(hmax, axis=0).argmax(axis=0) - 1  # 找到最后一个有特征的列号

    bound = int(min(h, w) * 0.01)  # 划线粗细
    top = np.clip(top, 0, h - bound)
    image[top:top + bound, :] = [255, 0, 0]  # 上限线
    bottom = np.clip(bottom, 0 + bound, h)
    image[bottom - bound:bottom, :] = [255, 0, 0]  # 下线
    left = np.clip(left, 0, w - bound)
    image[:, left:left + bound] = [255, 0, 0]  # 左线
    right = np.clip(right, 0 + bound, w)
    image[:, right - bound: right] = [255, 0, 0]  # 右线

    label = LaneSegDataset.decode_rgb(label)  # [H,W,C=3],ndarray RGB
    mask = mask[..., np.newaxis]  # [H,W,C=1],ndarray Alpha
    label = np.append(label, mask, axis=2)  # [H,W,C=4],ndarray 设置label的alpha通道

    image = Image.fromarray(image)
    label = Image.fromarray(label)

    fig, ax = plt.subplots(figsize=(20, 15))
    # ax.imshow(Image.blend(image, label, alpha=alpha))
    ax.imshow(Image.alpha_composite(image.convert('RGBA'),
                                    label.convert('RGBA')))
    ax.text(w, top, str(top), fontsize=16)
    ax.text(w, bottom, str(bottom), fontsize=16)
    ax.text(left, 0, str(left), fontsize=16)
    ax.text(right, 0, str(right), fontsize=16)

    plt.savefig('/root/private/imfolder/observe_' +
                time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime()) + name)  # 保存融合图像
    plt.close(fig)
    pass


if __name__ == '__main__':
    imdir = ('/root/data/LaneSeg/Image_Data/'
             'Road02/Record001/Camera 5')
    lbdir = ('/root/data/LaneSeg/Gray_Label/'
             'Label_road02/Label/Record001/Camera 5')

    for i, fim in enumerate(os.listdir(imdir)):
        # s = input('>>>')
        # if s == 'q':
        #     break
        #     pass
        if i > 220:  # 要多少张图片？220张
            break
            pass
        print(i, fim)
        flb = fim.replace('.jpg', '_bin.png')

        absim = os.path.join(imdir, fim)
        abslb = os.path.join(lbdir, flb)

        if os.path.exists(absim) and os.path.exists(abslb):
            im = Image.open(absim)
            lb = Image.open(abslb)
            observe_data(im, lb, name=fim)
            # predict(im, lb)
        pass
    pass
