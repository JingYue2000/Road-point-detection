#coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pprint
import os

config = dict()
# train hyper params config
config['lr'] = 1e-5
config['momentum'] = 0.95
config['weight_decay'] = 0.0005
config['epoch_num'] = 100
config['batch_size'] = 4
config['sigma'] = 1.3
# model config
config['point_num'] = 8
config['iters'] = 3
config['fusion_mode'] = 'cat' 
config['w_part'] = True
config['stride'] = 4  # (N, C, 256, 256) -> (N, C, 64, 64)
# train config
config['debug_vis'] = False
config['is_test'] = False
config['debug'] = True
config['fname'] = '/data2/data1/zc12345/private_datasets/tsd-max/tsd_max_scripts/train-{}.mat'.format(config['point_num'])
config['save_dir'] = './models/{}-points_{}-iters_{}-part_{}-fusion_models'.format(
                      config['point_num'], config['iters'], config['w_part'], config['fusion_mode'])
config['save_freq'] = 20
config['checkout'] = ''
config['start_epoch'] = 0
config['eval_freq'] = 10

def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,C,H,W)
    :return:numpy array (N,K,2)
    """
    # print('='*100)
    # print('heatmaps.shape:')
    # print(heatmaps.shape)
    N,C,H,W = heatmaps.shape     #(4, 31, 64, 64)
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            # print('='*100)
            # print('get_peak_points(heatmaps):')
            #print(heatmaps[i, j])
            #print(heatmaps[i, j].max())
            
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            # print(y)
            # print(x)
            # exit()
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)    # all_peak_points.shape=(4, 31, 2)
    # print(all_peak_points.shape)
    # exit()
    return all_peak_points

def get_mse(pred_points,gts):
    """
    :param pred_points: numpy (N,K,2)
    :param gts: numpy (N,K,2)
    :return:
    """
    #pred_points = Variable(torch.from_numpy(pred_points).float()).to(device)
    #gts = Variable(torch.from_numpy(gts).float()).to(device)
    criterion = nn.MSELoss()
    loss = criterion(pred_points,gts)
    return loss