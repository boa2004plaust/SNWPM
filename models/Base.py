# -*- coding: utf-8 -*-
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                    nn.BatchNorm2d(16),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(16),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                  )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                  )
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                  )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                  )
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                  )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.fc_1 = nn.Linear(256, 2)

    def forward(self, x):
        # x: #[none, 2, 72, 256]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = self.Flatten(x)
        out = self.fc_1(x)
        return out


