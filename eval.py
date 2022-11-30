# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:45:29 2021

@author: jingyue zhang
@contact: zhangjingyuezjy@163.com

@description:
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from data_loader import KFDataset
from model import keypointNet, BasicRPNet
from utils.util import get_peak_points, get_mse, config
from utils.seg_metrics import _fast_hist
from utils.keypoint_metrics import l2dist1, l2dist2, array2kpts
  
def evaluate(net, criterion, dataLoader, device):
    with torch.no_grad():
        net.eval()
        n_class = 2 # binary segmentation
        hist = np.zeros((n_class, n_class))
        sparse_dists = []
        dense_dists = []
        total_loss = []
        total_loss_coor = []

        #pbar = tqdm(total=len(dataLoader))
        #pbar.set_description("Evaluation")
        for i, (images, heatmaps, gt_points, gt_segs) in enumerate(dataLoader):
            #print(gt_points.shape)   # torch.Size([4, 16, 2])
            #print(gt_points)
            #print(gt_segs.shape)     # torch.Size([4, 256, 256])
            #exit()

            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs  = net.forward(images)
            pred_heatmaps = outputs[:,:2*config['point_num']]
            pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,K,2)
            #pred_points = pred_points.reshape((pred_points.shape[0],-1)) #(N,K*2)
            pred_points = pred_points * config['stride']
            loss_coor = get_mse(torch.from_numpy(pred_points).float().to(device), gt_points.float().to(device))
            loss = criterion(outputs, heatmaps)

            total_loss.append(loss.item())
            total_loss_coor.append(loss_coor.item())

            # keypoint metrics
            for lt, lp in zip(gt_points.numpy(), pred_points):
                sparse_dist = l2dist1(lt, lp)
                dense_dist = l2dist2(lt, lp)

                sparse_dists.append(sparse_dist)
                dense_dists.append(dense_dist)

            # get segmentation confusion matrix
            h, w = gt_segs.shape[-2], gt_segs.shape[-1] 
            coord1, coord2 = (w-1, h-1), (0, h-1)
            for lt, lp in zip(gt_segs, pred_points):
                img1 = Image.new('L', (w, h), 0)
                lp = array2kpts(lp)
                coord = list([coord1, coord2])
                vertices = lp + coord
                ImageDraw.Draw(img1).polygon(vertices, outline=1, fill=1)
                mask = np.array(img1)
                # plt.imshow(mask)
                # plt.show()
                lt = lt.numpy()
                hist += _fast_hist(lt.flatten(), mask.flatten(), n_class)

            #pbar.update(1)
        #pbar.close()
    # keypoint metrics
    mean_sparse_dist = np.mean(sparse_dists)
    mean_dense_dist = np.mean(dense_dists)
    
    # seg metrics
    overall_acc = np.diag(hist).sum() / hist.sum() # overall acc
    acc = np.diag(hist) / hist.sum(axis=1) # acc for each cls
    mean_acc = np.nanmean(acc) # mean acc for all cls
    precision = np.diag(hist) / hist.sum(axis=0) # prec for each cls
    mean_precision = np.nanmean(precision) # mean prec for all cls
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) # IoU for each cls
    mean_iu = np.nanmean(iu) # mean IoU
    freq = hist.sum(axis=1) / hist.sum() # frequency
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()#frequency weighted IoU

    return (total_loss, total_loss_coor), (overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc, mean_sparse_dist, mean_dense_dist)

def main():
    config['fname'] = '/data2/data1/zc12345/private_datasets/tsd-max/test-{}.mat'.format(config['point_num'])
    config['checkout'] = '{}/best_model.ckpt'.format(config['save_dir'])
    config['is_test'] = True
    config['debug_vis'] = False
    config['debug'] = True
    logging.basicConfig(level=logging.DEBUG,
                        filename='best_ckpt_summary_eval.log',
                        format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = BasicRPNet(
              outNode=2*config['point_num'], iters=config['iters'], fusion_mode=config['fusion_mode'], w_part=config['w_part'])
    net.float().to(device)
    criterion = torch.nn.MSELoss()
    if (config['checkout'] != ''):
        #net.load_state_dict(torch.load(config['checkout']))
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(config['checkout']).items()})

    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset,config['batch_size'],shuffle=False)
    print('evaluate start')
    loss, metrics = evaluate(net, criterion, dataLoader, device)
    print('evaluate end')
    loss, loss_coor = np.nanmean(loss[0]), np.nanmean(loss[1])
    overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc, mean_sparse_dist, mean_dense_dist = metrics

    model_checkout = config['checkout'].split('/')
    model_name = model_checkout[-2]
    
    print('\n model_name: {}\n \
          loss: {:15} loss_coor: {:15}\n \
          overall_acc: {:15}\n \
          acc_0: {:15} acc_1: {:15} mean_acc: {:15}\n \
          prec_0: {:15} prec_1: {:15} mean_prec: {:15}\n \
          iu_0: {:15} iu_1: {:15} mean_iu: {:15}\n \
          fwavacc: {:15}\n \
          mean_sparse_dist: {:15} mean_dense_dist: {:15}'.format(
        model_name, loss, loss_coor,
        overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc, 
        mean_sparse_dist, mean_dense_dist))
    logging.info('model_name: {} \
          loss: {:15} loss_coor: {:15} \
          overall_acc: {:15} \
          acc_0: {:15} acc_1: {:15} mean_acc: {:15} \
          prec_0: {:15} prec_1: {:15} mean_prec: {:15} \
          iu_0: {:15} iu_1: {:15} mean_iu: {:15} \
          fwavacc: {:15} \
          mean_sparse_dist: {:15} mean_dense_dist: {:15}'.format(
        model_name, loss, loss_coor,
        overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc, 
        mean_sparse_dist, mean_dense_dist))
    
if __name__ == '__main__':
    main()
