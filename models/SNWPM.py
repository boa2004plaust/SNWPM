# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv_fuse = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv_fuse(x)
        return self.sigmoid(x)


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
        self.ca = ChannelAttention(256)
        self.sa = SpatialAttention()

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
        
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.gap(x)
        x = self.Flatten(x)
        out = self.fc_1(x)
        return out


