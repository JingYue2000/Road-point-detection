#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Upsample
from torch.autograd import Variable
from hg import HourGlass, Lin

class KFSGNet(nn.Module):

    def __init__(self):
        super(KFSGNet,self).__init__()
        self.__conv1 = nn.Conv2d(3,64,1)
        self.__relu1 = nn.ReLU(inplace=True)
        self.__conv2 = nn.Conv2d(64,128,1)
        self.__relu2 = nn.ReLU(inplace=True)
        self.__hg = HourGlass(n=4,f=128)
        self.__lin = Lin(numIn=128,numout=15)
    def forward(self,x):
        x = self.__relu1(self.__conv1(x))
        x = self.__relu2(self.__conv2(x))
        x = self.__hg(x)
        x = self.__lin(x)
        return x


from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim

class tempDataset(Dataset):
    def __init__(self):
        self.X = np.random.randn(100,3,96,96)
        self.Y = np.random.randn(100,30,96,96)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        # do not set batch_size
        return self.X[item],self.Y[item]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = Variable(torch.rand(4, 3, 224, 224)).to(device)
    model = KFSGNet().to(device)
    out = model(x)
    print(out.shape)
    '''
    from torch.nn import MSELoss
    critical = MSELoss()

    dataset = tempDataset()
    dataLoader = DataLoader(dataset=dataset,batch_size=64)
    shg = KFSGNet().cuda()
    optimizer = optim.SGD(shg.parameters(), lr=0.001, momentum=0.9,weight_decay=1e-4)

    for e in range(200):
        for i,(x,y) in enumerate(dataLoader):
            x = Variable(x,requires_grad=True).float().cuda()
            y = Variable(y).float().cuda()
            y_pred = shg.forward(x)
            loss = critical(y_pred[0],y[0])
            print('loss : {}'.format(loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    '''