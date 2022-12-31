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
from models.SRPM import Model
from utils import Logger, create_exp_dir


class MyTrainset(Dataset):
    def __init__(self, trainX, trainY, trainX_aug1=None, trainX_aug2=None):
        self.x = trainX.astype(np.float32)
        self.y = trainY.astype(np.float32)
        if trainX_aug1 is not None:
            self.x = np.concatenate((self.x, trainX_aug1.astype(np.float32)), axis=0)
            self.y = np.concatenate((self.y, trainY.astype(np.float32)), axis=0)
        if trainX_aug2 is not None:
            self.x = np.concatenate((self.x, trainX_aug2.astype(np.float32)), axis=0)
            self.y = np.concatenate((self.y, trainY.astype(np.float32)), axis=0)

        self.len = len(self.y)

        print('TrainX data shape:', self.x.shape)
        print('TrainY data shape:', self.y.shape)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)


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


def train(idx, epoch, model, train_loader, optimizer, scheduler):
    model.train()
    loss_avg = 0
    for i, (x, y) in enumerate(train_loader):
        x = x.float().cuda()
        y = y.float().cuda()

        # 清零
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

    loss_avg /= len(train_loader)

    scheduler.step()

    lr = optimizer.param_groups[0]['lr']
    print('Run %d | Epoch : %d/%d, lr: %.6f, loss_avg: %.6f' % (
        idx, epoch + 1, cfg.epoch, lr, loss_avg))
    return loss_avg


if __name__ == '__main__':
    cfg = parser_args()
    cfg.exp_dir = cfg.exp_dir + '_' + cfg.arch + \
                  '_epoch'+str(cfg.epoch) + \
                  '_lr' + str(cfg.lr) + \
                  '_bs' + str(cfg.bs)
    print(cfg)
    setup_seed(2020)

    create_exp_dir(cfg.exp_dir, scripts_to_save=None)
    sys.stdout = Logger(os.path.join(cfg.exp_dir, 'log_train_' + cfg.name + '.txt'))

    trainX, trainY, testX, testY = data_read()
    trainX_aug1, testX_aug1 = data_preproc_aug1(trainX, testX)
    trainX_aug2, testX_aug2 = data_preproc_aug2(trainX, testX)

    if cfg.mode == 'orig':
        train_dataset = MyTrainset(trainX, trainY)
    elif cfg.mode == 'aug1':
        train_dataset = MyTrainset(trainX, trainY, trainX_aug1=trainX_aug1)
    elif cfg.mode == 'aug2':
        train_dataset = MyTrainset(trainX, trainY, trainX_aug2=trainX_aug2)
    elif cfg.mode == 'aug12':
        train_dataset = MyTrainset(trainX, trainY, trainX_aug1=trainX_aug1, trainX_aug2=trainX_aug2)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.bs, num_workers=cfg.num_works, shuffle=True)  # shuffle 标识要打乱顺序

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
        model = model.cuda()
        criterion = nn.MSELoss().cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epoch)

        test_MRSE_orig, test_CDF90_orig = 10000000, 10000000
        test_MRSE_aug1, test_CDF90_aug1 = 0, 0
        test_MRSE_aug2, test_CDF90_aug2 = 0, 0
        for epoch in range(cfg.epoch):
            train(idx, epoch, model, train_loader, optimizer, scheduler)
            test_MRSE, test_CDF90 = test(idx, epoch, model, test_loader, 'Test Orig')
            if test_MRSE < test_MRSE_orig:
                test_MRSE_orig, test_CDF90_orig = test_MRSE, test_CDF90
                test_MRSE_aug1, test_CDF90_aug1 = test(idx, epoch, model, test_loader_aug1, 'Test Aug1')
                test_MRSE_aug2, test_CDF90_aug2 = test(idx, epoch, model, test_loader_aug2, 'Test Aug2')
                print('### Model saved! Best test MRSE: %.6f.' % test_MRSE)
                torch.save(model.state_dict(), model_save)

        mean_Perf.append([test_MRSE_orig, test_CDF90_orig,
                          test_MRSE_aug1, test_CDF90_aug1,
                          test_MRSE_aug2, test_CDF90_aug2])

    print('\nPerformance:')
    print('MRSE_orig,CDF90_orig,MRSE_aug1,CDF90_aug1,MRSE_aug2,CDF90_aug2')
    for data in mean_Perf:
        for x in data:
            print('{:.4f}'.format(x), end=',')
        print('')
