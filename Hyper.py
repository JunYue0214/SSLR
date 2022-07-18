# -*- coding: utf-8 -*-
"""
@author: Vision-Zhu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import concatenate, Dense, Add
from keras.layers import Conv2D, Input, Activation, BatchNormalization
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Reshape

colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                   [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                   [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
                   [0, 0, 128]])


def resnet99_avg_recon(band1, band2, imx, ncla1, l=1):
    input1 = Input(shape=(imx, imx, band1))
    input2 = Input(shape=(imx, imx, band2))

    conv0x = Conv2D(32, kernel_size=(3, 3), padding='valid',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv0 = Conv2D(32, kernel_size=(3, 3), padding='valid',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn11 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros',
                              moving_variance_initializer='ones')
    conv11 = Conv2D(64, kernel_size=(3, 3), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv12 = Conv2D(64, kernel_size=(3, 3), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    bn21 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                              beta_initializer='zeros', gamma_initializer='ones',
                              moving_mean_initializer='zeros',
                              moving_variance_initializer='ones')
    conv21 = Conv2D(64, kernel_size=(3, 3), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    conv22 = Conv2D(64, kernel_size=(3, 3), padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    fc1 = Dense(ncla1, activation='softmax', name='output1',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    #
    dconv1 = Conv2DTranspose(64, kernel_size=(1, 1), padding='valid')
    dconv2 = Conv2DTranspose(64, kernel_size=(3, 3), padding='valid')
    dconv3 = Conv2DTranspose(64, kernel_size=(3, 3), padding='valid')
    dconv4 = Conv2DTranspose(64, kernel_size=(3, 3), padding='valid')
    dconv5 = Conv2DTranspose(band1, kernel_size=(3, 3), padding='valid')
    bn1_de = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')
    bn2_de = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                beta_initializer='zeros', gamma_initializer='ones',
                                moving_mean_initializer='zeros',
                                moving_variance_initializer='ones')

    # x1
    x1 = conv0(input1)
    x1x = conv0x(input2)
    x1 = concatenate([x1, x1x], axis=-1)
    x11 = bn11(x1)
    x11 = Activation('relu')(x11)
    x11 = conv11(x11)
    x11 = Activation('relu')(x11)
    x11 = conv12(x11)
    x1 = Add()([x1, x11])

    if l == 2:
        x11 = bn21(x1)
        x11 = Activation('relu')(x11)
        x11 = conv21(x11)
        x11 = Activation('relu')(x11)
        x11 = conv22(x11)
        x1 = Add()([x1, x11])

    x1 = GlobalAveragePooling2D(name='ploss')(x1)
    pre1 = fc1(x1)

    #
    x12 = Reshape((1, 1, 64))(x1)
    x12 = dconv1(x12)
    x12 = bn1_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv2(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv3(x12)
    x12 = bn2_de(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv4(x12)
    x12 = Activation('relu')(x12)
    x12 = dconv5(x12)

    model1 = Model(inputs=[input1, input2], outputs=[pre1, x12])
    model2 = Model(inputs=[input1, input2], outputs=pre1)
    return model1, model2


def imgDraw(label, imgName, path='./pictures', show=True):
    """
    功能：根据标签绘制RGB图
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，label有效范围[1,num]
    """
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  # 对背景设置为黑色
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return Y_RGB


def displayClassTable(n_list, matTitle=""):
    """
    功能：打印list的各元素
    输入：（list）
    输出：无
    备注：无
    """
    from pandas import DataFrame
    print("\n+--------- 原始输入数据" + matTitle + "统计结果 ------------+")
    lenth = len(n_list)  # 一共n个分类
    column = range(1, lenth + 1)
    table = {'Class': column, 'Total': [int(i) for i in n_list]}
    table_df = DataFrame(table).to_string(index=False)
    print(table_df)
    print('All available data total ' + str(int(sum(n_list))))
    print("+---------------------------------------------------+")


def listClassification(Y, matTitle=''):
    """
    功能：对标签数据计数并打印
    输入：（原始标签数据，是否打印）
    输出：分类结果
    备注：无
    """
    numClass = np.max(Y)  # 获取分类数
    listClass = []  # 用列表依次存储各类别的数量
    for i in range(numClass):
        listClass.append(len(np.where(Y == (i + 1))[0]))
    displayClassTable(listClass, matTitle)
    return listClass
