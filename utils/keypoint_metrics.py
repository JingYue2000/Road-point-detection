# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:56:00 2021

@author: jingyue zhang
@contact: zhangjingyuezjy@163.com

@description:
"""
 
from PIL import Image, ImageDraw
import numpy as np
import os.path as osp
import json
import logging

# set right bottom and left bottom coordinates
coord1, coord2, delta = (1279, 1023), (0, 1023), 5
h, w = 1024, 1280

def calc_iou(vertices1, vertices2, h ,w):
    '''
    https://github.com/AlexMa011/pytorch-polygon-rnn/blob/master/utils/utils.py
    calculate IoU of two polygons
    :param vertices1: vertices of the first polygon, tuple list
    :param vertices2: vertices of the second polygon, tuple list
    :return: the iou, the intersection area, the union area
    '''
    img1 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
    mask1 = np.array(img1)
    img2 = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
    mask2 = np.array(img2)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    nu = np.sum(intersection)
    de = np.sum(union)
    if de!=0:
        return nu*1.0/de, nu, de
    else:
        return 0, nu, de

def l2dist1(pts1, pts2):
    '''
    cacluate L2 distance
    '''
    assert pts1.shape == pts2.shape
    m, n = pts1.shape
    dist = np.linalg.norm(pts1 - pts2, axis=1, ord=2)
    return np.mean(dist)

def l2dist2(pts1, pts2):
    '''
    interp to dense points and cacluate L2 distance
    @params: pts1: ground truth points (numpy array)
    @params: pts2: prediction points (numpy array)
    @return: mean distance
    '''
    assert pts1.shape == pts2.shape
    m, n = pts1.shape
    pts1 = _interp(pts1)
    dist = []
    for pt in pts2:
        d = np.min(np.linalg.norm(pts1 - pt, axis=1, ord=2))
        dist.append(d)
    return np.mean(dist)

def _interp(pts1):
    # print(pts1[0, 0])
    # print(pts1[-1, 0])
    # print(pts1)
    _x1 = np.arange(pts1[0, 0], pts1[-1, 0] + 1, step=1)
    _y1 = np.interp(_x1, pts1[:, 0], pts1[:, 1])
    # print(_x1)
    # print(_y1)
    # exit()
    _x1, _y1 = np.trunc(_x1), np.trunc(_y1)
    pts1 = np.array([_x1, _y1]).T
    return pts1

def load_pts(file_path, type=None):
    ext = file_path.split('.')[-1]
    points = None
    if ext == 'json':
        data = json.load(open(file_path))
        if 'points' in data.keys():
            points = np.array(data['points'])
        elif 'version' in data.keys():
            for shape in data['shapes']:
                if shape['label'] == 'road':
                    points = np.array(shape['points'])
    elif ext == 'txt':
        points = np.loadtxt(file_path)
    else:
        logging.error('{} Unsolved File Extension: {}.'.format(file_path, ext))
    if points is not None and type is None:
        points = array2kpts(points)
    return points

def array2kpts(pts):
    pts = np.clip(pts, (0,0), coord1)
    np.place(pts[:,0], pts[:,0] < delta, 0)
    np.place(pts[:,0], coord1[0] - pts[:,0] < delta, coord1[0])
    np.place(pts[:,1], coord1[1] - pts[:,1] < delta, coord1[1])
    kpts = []
    for point in pts:
        kpts.append(tuple(point))
    return kpts

def test():
    data_dir = '../../../data/tsd-max-traffic'
    file_path1 = osp.join(data_dir, 'sequence-1', 'label', 'Section14CameraC_00736c.json')
    file_path2 = osp.join(data_dir, 'val-7pts-epoch50', 'Section5CameraC_00208c.txt')
    pts1, pts2 = load_pts(file_path1), load_pts(file_path2)
    coord = list([coord1, coord2])
    pts2 = pts2 + coord
    iou, nu, de = calc_iou(pts1, pts2, h=1024, w=1280)
    print(iou, nu, de)
    file_path1 = osp.join(data_dir, 'val-7pts-epoch50', 'Section14CameraC_00400c.txt')
    file_path2 = osp.join(data_dir, 'val-7pts-epoch50', 'Section14CameraC_00520c.txt')
    kpt_label_pts, pred_pts = load_pts(file_path1, type='np'), load_pts(file_path2, type='np')
    dist1 = l2dist1(kpt_label_pts, pred_pts)
    dist2 = l2dist2(kpt_label_pts, pred_pts)
    print(dist1, dist2)

if __name__ == '__main__':
    test()