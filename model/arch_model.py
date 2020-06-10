from __future__ import print_function

import torch
import torch.nn as nn


class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()

        self.linear1 = nn.Linear(12544 + 3136, 2048) # +12544
        self.linear2 = nn.Linear(2048, nb_classes)
        self.sigm = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x1 = self.modelA(x.clone())
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)

        out1 = self.lrelu(self.linear1(x))
        out2 = self.sigm(self.linear2(out1))
        return out2


class ConvNet_RN101(nn.Module):
    def __init__(self):
        super(ConvNet_RN101, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(2048, 1024, (5, 5), (1, 1), (2, 2))
        self.bn_1 = nn.BatchNorm2d(1024, eps=1e-5, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(1024, 512, (3, 3), (1, 1), (1, 1))
        self.bn_2 = nn.BatchNorm2d(512, eps=1e-5, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1))
        self.bn_3 = nn.BatchNorm2d(256, eps=1e-5, affine=True, track_running_stats=True)

        #self.linear1 = nn.Linear(in_features=12544, out_features=1024)
        #self.linear2 = nn.Linear(in_features=1024, out_features=101)

    def forward(self, x):
        x = self.lrelu(x)
        x = self.lrelu(self.bn_1(self.conv1(x)))
        x = self.lrelu(self.bn_2(self.conv2(x)))
        x = self.lrelu(self.bn_3(self.conv3(x)))

        fea = x.view(x.size(0), -1)
        #out1 = self.lrelu(self.linear1(fea))
        #out2 = self.lrelu(self.linear2(out1))

        return fea


class ConvNet_RN34(nn.Module):
    def __init__(self):
        super(ConvNet_RN34, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(512, 256, (5, 5), (1, 1), (2, 2))
        self.bn_1 = nn.BatchNorm2d(256, eps=1e-5, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1))
        self.bn_2 = nn.BatchNorm2d(128, eps=1e-5, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.bn_3 = nn.BatchNorm2d(64, eps=1e-5, affine=True, track_running_stats=True)

        #self.linear1 = nn.Linear(in_features=12544, out_features=1024)
        #self.linear2 = nn.Linear(in_features=1024, out_features=101)

    def forward(self, x):
        x = self.lrelu(x)
        x = self.lrelu(self.bn_1(self.conv1(x)))
        x = self.lrelu(self.bn_2(self.conv2(x)))
        x = self.lrelu(self.bn_3(self.conv3(x)))

        fea = x.view(x.size(0), -1)
        #out1 = self.lrelu(self.linear1(fea))
        #out2 = self.lrelu(self.linear2(out1))

        return fea


class MyFC_RN101(nn.Module):
    def __init__(self):
        super(MyFC_RN101, self).__init__()

        self.linear1 = nn.Linear(in_features=12544, out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=101)
        self.dropout = nn.Dropout(0.4)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out1 = self.lrelu(self.linear1(x))
        out2 = self.dropout(out1)
        out3 = self.sigm(self.linear2(out2))

        return out3


class MyFC_RN34(nn.Module):
    def __init__(self):
        super(MyFC_RN34, self).__init__()

        self.linear1 = nn.Linear(in_features=3136, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=101)
        self.dropout = nn.Dropout(0.4)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        out1 = self.lrelu(self.linear1(x))
        out2 = self.dropout(out1)
        out3 = self.sigm(self.linear2(out2))

        return out3