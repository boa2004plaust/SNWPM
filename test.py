import os
import sys
# import h5py
# import glob
import random
import argparse
import numpy as np
from copy import deepcopy as DP

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.model import Model
from utils import Logger, create_exp_dir


class MyTestset(Dataset):
    def __init__(self, dataX, dataY):
        self.x = dataX.astype(np.float32)
        self.y = dataY.astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        
        return (x, y)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='log')
    parser.add_argument('--arch', type=str, default='SRPM10', help='[SRPM5, SRPM10, SRPM15]')
    parser.add_argument('--name', type=str, default='using_orig')
    parser.add_argument('--mode', type=str, default='orig')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--num_works', type=int, default=4)
    #parser.add_argument('--gamma', type=float, default=0.1, help='')
    #parser.add_argument('--n_steps', type=int, default=40, help='number of epochs to update learning rate')
    return parser.parse_args()


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_read():
    file_name1 = '../../data/raw_data/Case_1_2_Training.npy'
    print('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    print('orignal training data', CIR.shape)
    Xtrain = CIR.transpose((2, 0, 3, 1))  # [none, 2, 72, 256]
    trainX = Xtrain[:10000, :, :, :]
    testX = Xtrain[10000:, :, :, :]
    print('Xtrain data', Xtrain.shape)

    file_name2 = '../../data/raw_data/Case_1_2_Training_Label.npy'
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


def test(idx, epoch, model, test_loader, info):
    model.eval()
    py_list, sy_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.float().cuda()
            y = y.float().cuda()
            output = model(x)
            py_list.append(output.detach())
            sy_list.append(y.detach())

    y_test = torch.cat(py_list, dim=0).cpu().data.numpy()
    y = torch.cat(sy_list, dim=0).cpu().data.numpy()
    Diff = np.sqrt(np.sum((np.square(y_test - y)), 1))  # Euclidean distance
    Order = np.sort(Diff)
    Score_CDF90 = Order[int(y_test.shape[0] * 0.9)]
    Score_MRSE = np.mean(Order)

    print('Run %d | %s Epoch : %d/%d, Score_CDF90: %.6f, Score_MRSE: %.6f' % (
        idx, info, epoch + 1, cfg.epoch, Score_CDF90, Score_MRSE))
    return Score_MRSE, Score_CDF90


if __name__ == '__main__':
    cfg = parser_args()
    cfg.exp_dir = cfg.exp_dir + '_' + cfg.arch + \
                  '_epoch'+str(cfg.epoch) + \
                  '_lr' + str(cfg.lr) + \
                  '_bs' + str(cfg.bs)
    print(cfg)
    setup_seed(2020)

    create_exp_dir(cfg.exp_dir, scripts_to_save=None)
    sys.stdout = Logger(os.path.join(cfg.exp_dir, 'log_test_' + cfg.name + '.txt'))

    trainX, trainY, testX, testY = data_read()
    trainX_aug1, testX_aug1 = data_preproc_aug1(trainX, testX)
    trainX_aug2, testX_aug2 = data_preproc_aug2(trainX, testX)

    test_dataset = MyTestset(testX, testY)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg.bs, num_workers=cfg.num_works)
    test_dataset_aug1 = MyTestset(testX_aug1, testY)
    test_loader_aug1 = DataLoader(dataset=test_dataset_aug1, batch_size=cfg.bs, num_workers=cfg.num_works)
    test_dataset_aug2 = MyTestset(testX_aug2, testY)
    test_loader_aug2 = DataLoader(dataset=test_dataset_aug2, batch_size=cfg.bs, num_workers=cfg.num_works)

    mean_Perf = []
    for idx in range(5):
        setup_seed(2020+idx*3000)

        model_save = os.path.join(cfg.exp_dir, 'model_'+cfg.name+'_'+str(idx+1)+'.pth')

        model = Model()
        model_state_dict = torch.load(model_save, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
        model = model.cuda()

        test_MRSE_orig, test_CDF90_orig = test(idx, cfg.epoch, model, test_loader, 'Test Orig')
        test_MRSE_aug1, test_CDF90_aug1 = test(idx, cfg.epoch, model, test_loader_aug1, 'Test Aug1')
        test_MRSE_aug2, test_CDF90_aug2 = test(idx, cfg.epoch, model, test_loader_aug2, 'Test Aug2')

        mean_Perf.append([test_MRSE_orig, test_CDF90_orig,
                          test_MRSE_aug1, test_CDF90_aug1,
                          test_MRSE_aug2, test_CDF90_aug2])

    print('\nPerformance:')
    print('MRSE_orig,CDF90_orig,MRSE_aug1,CDF90_aug1,MRSE_aug2,CDF90_aug2')
    for data in mean_Perf:
        for x in data:
            print('{:.4f}'.format(x), end=',')
        print('')
