#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from data_loader import KFDataset, transform
from model import keypointNet, BasicRPNet
from utils.util import get_peak_points, config

INDEX = 6

fnames = [
    'Section80CameraC_01260c',
    'Section80CameraC_00036c',
    'Section65CameraC_01115c',
    'Section63CameraC_00816c',
    'Section63CameraC_00000c',
    'Section49CameraC_01399c',
    'result_plot3']
config['fname'] = '/data3/2019/gjx/zjy/road_keypoint_detection/test-image/{}.jpg'.format(
    fnames[INDEX])
config['fname_seg'] = '/data3/2019/gjx/zjy/road_keypoint_detection/test-image/{}_green.png'.format(
    fnames[INDEX])
config['img_shape'] = np.array([256, 256])
config['is_test'] = True
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


def plot_demo(X, y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()


def test():
    with torch.no_grad():
        import time
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')
        net = BasicRPNet(
            outNode=2*config['point_num'], iters=config['iters'], fusion_mode=config['fusion_mode'], w_part=config['w_part'])
        net.float().to(device)
        net.eval()
        if (config['checkout'] != ''):
            net.load_state_dict(torch.load(config['checkout']))
            #net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(config['checkout']).items()})

        image = Image.open(config['fname'])
        width, height = image.size
        image = np.array(image.resize(config['img_shape']))
        image_seg = Image.open(config['fname_seg'])
        image_seg = np.array(image_seg)
        #image_seg = np.array(image_seg.resize(config['img_shape']))
        img = transform(image)
        img = torch.reshape(img, (1, 3, image.shape[0], image.shape[1]))
        img = img.float().to(device)
        print(img.shape)
        time_start = time.time()
        outputs = net.forward(img)
        pred_heatmaps = outputs[:, :2*config['point_num']]
        part_heatmaps = outputs[:, 2*config['point_num']:]
        #pred_heatmaps = net.forward(images)

        demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis, ...]
        demo_pred_poins = get_peak_points(demo_heatmaps)[0]  # (K,2)
        demo_pred_poins = demo_pred_poins * config['stride']  # (K,2)
        time_end = time.time()
        fig, ax = plt.subplots()
        plt.axis('off')
        #height, width = config['img_shape']
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.set_size_inches(width/100.0, height/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(image_seg)
        demo_pred_poins = demo_pred_poins * \
            np.array([width, height]) / config['img_shape']
        plt.scatter(demo_pred_poins[:, 0], demo_pred_poins[:, 1], c='red', s=50)
        plt.plot(demo_pred_poins[:, 0], demo_pred_poins[:, 1], c='red')
        plt.savefig('figures/{}_result.pdf'.format(fnames[INDEX]), dpi=300)
        plt.show()
        #plot_demo(demo_heatmaps.squeeze(), demo_pred_poins/config['stride'])

        print('totally cost', time_end-time_start)
        # print(demo_pred_poins)


if __name__ == '__main__':
    test()
