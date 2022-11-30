#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-5-13 13:23:43
# @Author  : jingyue zhang

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms

from data_loader import KFDataset
from model import keypointNet
from utils.util import get_peak_points, config

config['fname'] = '/data2/data1/zc12345/private_datasets/tsd-max/train-8.mat'
config['is_test'] = True
config['batch_size'] = 1
config['debug_vis'] = False
config['debug'] = True
config['checkout'] = '{}/best_model.ckpt'.format(config['save_dir'])

def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (15,2)
    :param axis:
    :return:
    """
    img = x.reshape(64, 64)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0], y[1], marker='x', s=10)

def plot_demo(X,y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()

def test():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    net = keypointNet(outNode=16)
    #net = KFSGNet(outNode=16)
    net.float().to(device)
    net.eval()
    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']),strict=False)
        # net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(config['checkout']).items()})

    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset,config['batch_size'],shuffle=False)
    all_result = []
    num = len(dataset)
    for i,(images,heatmaps,gts,gt_segs) in enumerate(dataLoader):
        print('{} / {}'.format(i,num))
        images = Variable(images).float().to(device)
        gts = gts.numpy()
        out4 = net.forward(images)
        pred_heatmaps = out4[:,:2*config['point_num']]
        part_heatmaps = out4[:,2*config['point_num']:]
        #pred_heatmaps = net.forward(images)

        demo_img = images[0].cpu().clone()
        unloader = transforms.ToPILImage()
        demo_img = unloader(demo_img)
        demo_gt = gts[0]
        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis,...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0] # (K,2)
        demo_pred_poins = demo_pred_poins * config['stride'] # (K,2)
        plt.imshow(demo_img)
        plt.scatter(demo_pred_poins[:,0],demo_pred_poins[:,1], c='red')
        plt.plot(demo_pred_poins[:,0],demo_pred_poins[:,1], c='red')
        plt.scatter(demo_gt[:,0],demo_gt[:,1], c='green')
        plt.show()
        #plot_demo(demo_heatmaps.squeeze(), demo_pred_poins/config['stride'])
        #plot_demo(heatmaps.squeeze(), demo_gt.squeeze()/config['stride'])

        pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,K,2)
        #pred_points = pred_points.reshape((pred_points.shape[0],-1)) #(N,K*2)
        print(pred_points[0,:,0],pred_points[0,:,1])
        
        # loss = get_mse(demo_pred_poins[np.newaxis,...],gts)
    
if __name__ == '__main__':
    test()
