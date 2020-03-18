import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def id_to_trainid(label_id):
    """
    要求输入的图像是灰度读图，灰阶0-255。
    matplotlib.img读取灰度图像的灰阶是0.0-1.0，需要自己处理到0-255
    :param label_id: id标注的label图像
    :return: trainid标注的label图像
    """
    label_trainid = np.zeros(label_id.shape, dtype=np.uint8)

    # trainId = 0
    # dst[src == 0] = 0
    label_trainid[label_id == 255] = 0
    label_trainid[label_id == 249] = 0

    # trainId = 1
    label_trainid[label_id == 200] = 1
    label_trainid[label_id == 204] = 1
    label_trainid[label_id == 213] = 0  # ignoreInEval
    label_trainid[label_id == 209] = 1
    label_trainid[label_id == 206] = 0
    label_trainid[label_id == 207] = 0

    # trainId = 2
    label_trainid[label_id == 201] = 2
    label_trainid[label_id == 203] = 2
    label_trainid[label_id == 211] = 0
    label_trainid[label_id == 208] = 0

    # trainId = 3
    label_trainid[label_id == 216] = 0
    label_trainid[label_id == 217] = 3
    label_trainid[label_id == 215] = 0

    # trainId = 4
    label_trainid[label_id == 218] = 0
    label_trainid[label_id == 219] = 0

    # trainId = 5->4,因trainId=4都被忽略，5递进为4，后面一样递进
    label_trainid[label_id == 210] = 4
    label_trainid[label_id == 232] = 0

    # trainId = 6->5
    label_trainid[label_id == 214] = 5

    # trainId = 7->6
    label_trainid[label_id == 202] = 0
    label_trainid[label_id == 220] = 6
    label_trainid[label_id == 221] = 6
    label_trainid[label_id == 222] = 6
    label_trainid[label_id == 231] = 0
    label_trainid[label_id == 224] = 6
    label_trainid[label_id == 225] = 6
    label_trainid[label_id == 226] = 6
    label_trainid[label_id == 230] = 0
    label_trainid[label_id == 228] = 0
    label_trainid[label_id == 229] = 0
    label_trainid[label_id == 233] = 0

    # trainId = 8->7
    label_trainid[label_id == 205] = 7
    label_trainid[label_id == 212] = 0
    label_trainid[label_id == 227] = 7
    label_trainid[label_id == 223] = 0
    label_trainid[label_id == 250] = 7

    return label_trainid


def trainid_to_id(label_trainid):
    """
    要求输入的图像是灰度读图，灰阶0-255。
    matplotlib.img读取灰度图像的灰阶是0.0-1.0，需要自己处理到0-255
    :param label_trainid: trainid标注的label图像
    :return: id标注的label图像
    """
    label_id = np.zeros(label_trainid.shape, dtype=np.uint8)

    # id = 0
    label_id[label_trainid == 0] = 0

    # id = 1
    label_id[label_trainid == 1] = 200

    # id = 2
    label_id[label_trainid == 2] = 201

    # id = 3
    label_id[label_trainid == 3] = 217

    # id = 4->5,因trainId=4都被忽略，5递进为4，转换为id需要处理，后面一样递进
    label_id[label_trainid == 4] = 210

    # id = 5->6
    label_id[label_trainid == 5] = 214

    # id = 6->7
    label_id[label_trainid == 6] = 228

    # id = 7->8
    label_id[label_trainid == 7] = 205

    return label_id


def gray_to_rgb(label_gray):
    """
    将trainid标注的灰度label转换成rgb彩色label
    :param label_gray: 灰度图label
    :return: rgb彩色label
    """
    height, width = label_gray.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    # id = 0
    rgb[label_gray == 0] = np.array([0, 0, 0])

    # id = 1
    rgb[label_gray == 1] = np.array([70, 130, 180])  # 浅蓝色

    # id = 2
    rgb[label_gray == 2] = np.array([0, 0, 142])  # 深蓝色

    # id = 3
    rgb[label_gray == 3] = np.array([220, 220, 0])  # 黄色

    # id = 4->5,因trainId=4都被忽略，5递进为4，转换为RGBlabel的时候要处理，后面一样递进
    rgb[label_gray == 4] = np.array([128, 64, 128])  # 紫色

    # id = 5->6
    rgb[label_gray == 5] = np.array([190, 153, 153])  # 浅粉色人行道

    # id = 6->7
    rgb[label_gray == 6] = np.array([51, 255, 51])  # 绿色

    # id = 7->8
    rgb[label_gray == 7] = np.array([255, 128, 0])  # 橘黄色

    return rgb


if __name__ == '__main__':
    """
    单元测试
    """
    file = '/root/data/LaneSeg/Gray_Label/Label_road02/Label/Record002/Camera 5/170927_064016919_Camera_5_bin.png'
    img = mpimg.imread(file)  # 读取一个gray_label图片
    img = img * 255  # 转换为0-255灰阶
    img = img.astype(np.uint8)  # 处理成0-255整数
    img_trainid = id_to_trainid(img)
    img_id = trainid_to_id(img_trainid)
    img_rgb = gray_to_rgb(img_trainid)


    def show(img1, img2, img3, img4):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        ax = ax.flatten()
        ax[0].imshow(img1, plt.cm.gray)  # 原始label图像，id标注
        ax[1].imshow(img2, plt.cm.gray)  # train id标注
        ax[2].imshow(img3, plt.cm.gray)  # 再转换为id标注，但一类别的只转换为一个id
        ax[3].imshow(img4)  # 再转换为RGB标注
        fig.tight_layout()
        plt.show()


    show(img, img_trainid, img_id, img_rgb)
