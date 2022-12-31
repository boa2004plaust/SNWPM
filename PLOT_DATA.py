import os
import sys
# import h5py
# import glob
import random
import argparse
import numpy as np
from copy import deepcopy as DP
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.cm as cm  # matplotlib内置的颜色地图
from mpl_toolkits.mplot3d import Axes3D  # 引入3d绘图模块


def plot_signal(data, saveName):
    data_matrix = np.multiply(data[0, :, :], data[0, :, :]) \
                  + np.multiply(data[1, :, :], data[1, :, :])
    data_matrix = np.sqrt(data_matrix)
    # print(data_matrix.shape)

    X = np.arange(0, 256, 1)
    Y = np.arange(0, 72, 1)
    X, Y = np.meshgrid(X, Y)  # 生成网格点矩阵，就是对X，Y进行网格化
    Z = data_matrix
    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    fig = plt.figure(figsize=(6, 6))  # 创建图片
    sub = fig.add_subplot(111, projection='3d')  # 添加子图，
    surf = sub.plot_surface(X, Y, Z, cmap=plt.cm.terrain)  # 绘制曲面,cmap=plt.cm.brg并设置颜色cmap
    # cb = fig.colorbar(surf, shrink=0.8, aspect=15)  # 设置颜色棒

    sub.set_xlabel('Subcarrier')
    sub.set_ylabel('Antenna')
    sub.set_zlabel('Amplitude')
    # plt.show()
    plt.savefig(saveName, format='png', dpi=300, bbox_inches="tight")


def plot_label(data, saveName):
    print(data.shape)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4, 2))  # 创建图片
    plt.scatter(data[0], data[1], s=30, marker='*', c='r')
    plt.xlim(0, 120)
    plt.ylim(0, 60)
    plt.grid(linestyle=':')
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(20))
    # plt.show()
    plt.savefig(saveName, format='png', dpi=300, bbox_inches="tight" )


def data_read():
    file_name1 = '../data/raw_data/Case_1_2_Training.npy'
    print('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    print('orignal training data', CIR.shape)
    Xtrain = CIR.transpose((2, 0, 3, 1))  # [none, 2, 72, 256]
    trainX = Xtrain[:10000, :, :, :]
    testX = Xtrain[10000:, :, :, :]
    print('Xtrain data', Xtrain.shape)

    file_name2 = '../data/raw_data/Case_1_2_Training_Label.npy'
    print('The current dataset is : %s' % (file_name2))
    POS = np.load(file_name2)
    print('orignal training label', POS.shape)
    Ytrain = POS.transpose((1, 0))  # [none, 2]
    trainY = Ytrain[:10000, :]
    testY = Ytrain[10000:, :]
    print('Ytrain data', Ytrain.shape)

    return trainX, trainY, testX, testY


def data_preproc_aug1(trainX, testX):
    trainX_aug1 = DP(trainX)
    trainX_aug1[:, :, 4:20, :] = 0
    trainX_aug1[:, :, 24:48, :] = 0
    trainX_aug1[:, :, 52:68, :] = 0
    testX_aug1 = DP(testX)
    testX_aug1[:, :, 4:20, :] = 0
    testX_aug1[:, :, 24:48, :] = 0
    testX_aug1[:, :, 52:68, :] = 0
    return trainX_aug1, testX_aug1


def data_preproc_aug2(trainX, testX):
    trainX_aug2 = DP(trainX)
    random.seed(0)
    for i in range(trainX_aug2.shape[0]):
        idx_bs = random.randint(0, 18)
        trainX_aug2[i, :, 4 * idx_bs:4 * idx_bs + 4, :] = 0
        if random.random() > 0.5:
            idx_bs = random.randint(0, 18)
            trainX_aug2[i, :, 4 * idx_bs:4 * idx_bs + 4, :] = 0
    testX_aug2 = DP(testX)
    for i in range(testX_aug2.shape[0]):
        idx_bs = random.randint(0, 18)
        testX_aug2[i, :, 4 * idx_bs:4 * idx_bs + 4, :] = 0
        if random.random() > 0.5:
            idx_bs = random.randint(0, 18)
            testX_aug2[i, :, 4 * idx_bs:4 * idx_bs + 4, :] = 0
    return trainX_aug2, testX_aug2


if __name__ == '__main__':
    trainX, trainY, testX, testY = data_read()
    trainX_aug1, testX_aug1 = data_preproc_aug1(trainX, testX)
    trainX_aug2, testX_aug2 = data_preproc_aug2(trainX, testX)

    data_idx = 512
    plot_signal(testX[data_idx, :, :, :], 'data_test_'+str(data_idx)+'.png')
    plot_label(testY[data_idx, :], 'label_test_'+str(data_idx)+'.png')
    plot_signal(testX_aug1[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug1.png')
    plot_signal(testX_aug2[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug2.png')

    data_idx = 1024
    plot_signal(testX[data_idx, :, :, :], 'data_test_'+str(data_idx)+'.png')
    plot_label(testY[data_idx, :], 'label_test_'+str(data_idx)+'.png')
    plot_signal(testX_aug1[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug1.png')
    plot_signal(testX_aug2[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug2.png')

    data_idx = 2048
    plot_signal(testX[data_idx, :, :, :], 'data_test_'+str(data_idx)+'.png')
    plot_label(testY[data_idx, :], 'label_test_'+str(data_idx)+'.png')
    plot_signal(testX_aug1[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug1.png')
    plot_signal(testX_aug2[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug2.png')

    data_idx = 4096
    plot_signal(testX[data_idx, :, :, :], 'data_test_'+str(data_idx)+'.png')
    plot_label(testY[data_idx, :], 'label_test_'+str(data_idx)+'.png')
    plot_signal(testX_aug1[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug1.png')
    plot_signal(testX_aug2[data_idx, :, :, :], 'data_test_'+str(data_idx)+'_aug2.png')

