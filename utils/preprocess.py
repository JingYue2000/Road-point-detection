#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File          :  preprocess.py
@Time          :  2021/01/08 09:52:26
@Author        :  jingyue
@Version       :  1.0
@Contact       :  zhangjingyuezjy@163.com
@Description   :  Preprocess images:
                    1. jpg2bmp
                    2. generate box
'''

# here put the import lib
import os
import math
import json
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt

def parse_lst(root_dir, box_dir, bmp_dir, lst_path):
    img_paths = []
    with open(lst_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_fn, json_fn = line.split()
            fn = img_fn.split('/')[-1]
            fn, _ext = fn.split('.')
            img_path = os.path.join(root_dir, img_fn)
            json_path = os.path.join(root_dir, json_fn)
            box_path = os.path.join(box_dir, fn+'_box.bmp')
            bmp_path = os.path.join(bmp_dir, fn+'.bmp')

            generate_box(json_path, box_path)
            jpg2bmp(img_path, bmp_path)

            print(img_fn)
    return

def generate_box(json_path, box_path):
    box_color, bg_color = 128, 0#np.array([128, 128, 128]), np.array([0, 0, 0])
    data = json.load(open(json_path))
    dshape = data['shapes']
    upper = 0
    for shape in dshape:
        if shape['label'] == 'road':
            points = shape['points']
            points = points / np.array([1280, 1024]) * np.array([256, 256])
            upper = np.min(points[:, 1])
            upper = math.floor(upper)
    box = np.zeros((256, 256, 3), dtype='uint8')
    box[:upper, :, :] = bg_color
    box[upper:, :, :] = box_color
    img = Image.fromarray(box)
    # plt.imshow(box)
    # plt.scatter(points[:,0], points[:,1])
    # plt.show()
    img.save(box_path)


def jpg2bmp(jpg_path, bmp_path):
    img = Image.open(jpg_path)
    img = img.resize((256, 256))
    img.save(bmp_path)

if __name__ == "__main__":
    root_dir = 'D:/documents/data/tsd-max-traffic'
    target_dir = 'D:/documents/data/high_order_energy'
    box_dir = os.path.join(target_dir, 'box')
    bmp_dir = os.path.join(target_dir, 'image')
    lst_path = os.path.join(root_dir, 'test_lst.txt')
    parse_lst(root_dir, box_dir, bmp_dir, lst_path)