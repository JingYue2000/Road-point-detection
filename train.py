#coding=utf-8
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import logging
import os
from tqdm import tqdm

from data_loader import KFDataset
from model import keypointNet, BasicRPNet
from eval import evaluate
from utils.util import get_peak_points, get_mse, config

if not os.path.exists(config['save_dir']):
    os.mkdir(config['save_dir'])

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def train(net, optimizer, criterion, trainDataLoader, device):
    logging.basicConfig(level=logging.DEBUG,
                    filename='{}/train.log'.format(config['save_dir']),
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')
    net.train()
    #pbar = tqdm(total=len(trainDataLoader))
    #pbar.set_description("Train")
    for i, (inputs, heatmaps_targets, gts, _) in enumerate(trainDataLoader):
        inputs = inputs.to(device)
        heatmaps_targets = heatmaps_targets.to(device)
        
        optimizer.zero_grad()
        #out1, out2, out3, out4 = net(inputs)
        outputs = net(inputs)
        #print(inputs.shape,outputs.shape, heatmaps_targets.shape)
        # eval
        pred_points = get_peak_points(outputs.cpu().data.numpy())
        pred_points = pred_points[:,:2*config['point_num']]
        pred_points = torch.from_numpy(pred_points*config['stride']).float().to(device)
        loss_coor = get_mse(pred_points, gts.float().to(device))
        loss = criterion(outputs, heatmaps_targets)
        # loss1 = criterion(out1, heatmaps_targets)
        # loss2 = criterion(out2, heatmaps_targets)
        # loss3 = criterion(out3, heatmaps_targets)
        # loss4 = criterion(out4, heatmaps_targets)
        
        # debug value
        #v_max = torch.max(outputs)
        #v_min = torch.min(outputs) # about [-1, 1]
        
        loss.backward()
        optimizer.step()
        #pbar.update(1)
    #pbar.close()
              
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    pprint.pprint(config)
    torch.manual_seed(0)
    cudnn.benchmark = True
    net = BasicRPNet(
              outNode=2*config['point_num'], iters=config['iters'], fusion_mode=config['fusion_mode'], w_part=config['w_part'])
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net, device_ids=[0,1])
    net.float().to(device)
    criterion = nn.MSELoss()
    #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    lr_init = optimizer.param_groups[0]['lr']
    trainDataset = KFDataset(config)
    trainDataset.load()
    trainDataLoader = DataLoader(trainDataset,config['batch_size'],True)
    sample_num = len(trainDataset)
    min_loss = np.inf

    if (config['checkout'] != ''):
        net.load_state_dict(torch.load(config['checkout']))
        print('load model from {}'.format(config['checkout']))

    for epoch in range(config['start_epoch'],config['epoch_num']+config['start_epoch']):

        #adjust_learning_rate(optimizer, epoch, lr_init)
        train(net, optimizer, criterion, trainDataLoader, device)
        loss, metrics = evaluate(net, criterion, trainDataLoader, device) # (overall_acc, acc, mean_acc, iu, mean_iu, fwavacc, mean_sparse_dist, mean_dense_dist)
        print('start epoch')
        loss, loss_coor = np.sum(loss[0]), np.sum(loss[1])
        overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc, mean_sparse_dist, mean_dense_dist = metrics
        lr = optimizer.param_groups[0]['lr']

        print('[Epoch {:002d}/{:002d}] -> lr: {} loss: {:15} loss_coor: {:15}\n \
              \t overall_acc: {:15}\n \
              \t acc_0: {:15} acc_1: {:15} mean_acc: {:15}\n \
              \t prec_0: {:15} prec_1: {:15} mean_prec: {:15}\n \
              \t iu_0: {:15} iu_1: {:15} mean_iu: {:15}\n \
              \t fwavacc: {:15}\n \
              \t mean_sparse_dist: {:15} mean_dense_dist: {:15}'.format(
            epoch + 1, config['epoch_num'], lr, loss, loss_coor,
            overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc, 
            mean_sparse_dist, mean_dense_dist))
        logging.info('[Epoch {:002d}/{:002d}] -> lr: {} loss: {:15} loss_coor: {:15} overall_acc: {:15} acc_0: {:15} acc_1: {:15} mean_acc: {:15} prec_0: {:15} prec_1: {:15} mean_prec: {:15} iu_0: {:15} iu_1: {:15} mean_iu: {:15} fwavacc: {:15} mean_sparse_dist: {:15} mean_dense_dist: {:15}'.format(
            epoch + 1, config['epoch_num'], lr, loss, loss_coor,
            overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc, 
            mean_sparse_dist, mean_dense_dist))
        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            torch.save(net.state_dict(),'{}/kd_epoch_{}_model.ckpt'.format(config['save_dir'], epoch+1))
            
        if loss < min_loss:
            min_loss = loss
            torch.save(net.state_dict(),'{}/best_model.ckpt'.format(config['save_dir']))

if __name__ == '__main__':
    main()