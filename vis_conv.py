import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import keypointNet
from data_loader import KFDataset
import matplotlib.pyplot as plt
from utils.util import config

def viz(module, input):
    x = input[0][0]
    x = x.cpu().numpy()
    min_num = np.minimum(16, x.shape[0])
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(min_num):
        fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        plt.imshow(x[i])
    plt.show()

config['is_test'] = True
config['debug_vis'] = False
config['debug'] = True
config['checkout'] = './models/kd_epoch_100_model.ckpt'

import cv2
import numpy as np
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = keypointNet(outNode=16).to(device)
    model.eval()
    #model.load_state_dict(torch.load(config['checkout']))
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(config['checkout']).items()})

    dataset = KFDataset(config)
    dataset.load()
    dataLoader = DataLoader(dataset,config['batch_size'])
    for i,(images,heatmaps,kpt_gts, seg_gts) in enumerate(dataLoader):
        for name, m in model.named_modules():
            # if not isinstance(m, torch.nn.ModuleList) and \
            #         not isinstance(m, torch.nn.Sequential) and \
            #         type(m) in torch.nn.__dict__.values():
            if isinstance(m, torch.nn.Conv2d):
                m.register_forward_pre_hook(viz)
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        with torch.no_grad():
            model(images)
        break

if __name__ == '__main__':
    main()