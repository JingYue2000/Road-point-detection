#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-7-13 19:32:16
# @Author  : jingyue zhang

# here put the import lib
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from seg_metrics import _fast_hist


def load_gt_img(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img = np.array(img)
    return img


def load_pred_img(img_path):
    img = Image.open(img_path)
    img = img.resize((256, 256))
    img = np.array(img)
    label_set = set(img.flatten())
    labels = sorted(label_set)
    if len(labels) == 2:
        img[img == labels[1]] = 1
    elif len(label_set) == 3:
        img[img == labels[1]] = 1
        img[img == labels[2]] = 2
    elif len(label_set) == 4:
        img[img == labels[1]] = 1
        img[img == labels[2]] = 2
        img[img == labels[3]] = 3
    else:
        raise ValueError
    return img


def load_pred_mat(mat_path):
    data = sio.loadmat(mat_path)
    img = data['L']
    lsp = data["Lsp"]
    llist = data['labelList']
    img = np.array(img)
    img[img == 1] = 0
    img[img == 2] = 1
    return img


def load_bmp_img(bmp_path):
    img = Image.open(bmp_path)
    img = img.convert('1')
    img = np.array(img)
    return img


def load_gt_mat(mat_path):
    data = sio.loadmat(mat_path)
    img = data['S']
    llist = data['names']
    img = np.array(img)
    img[img == 1] = 0
    img[img == 2] = 1
    return img


def png_eval(gt_dir, pred_dir, n_class=4):
    files = os.listdir(pred_dir)
    hist = np.zeros((n_class, n_class))
    for f in files:
        if os.path.splitext(f)[1] == '.jpg':
            gt_path = os.path.join(gt_dir, f[:-4]+'_label.png')
            pred_path = os.path.join(pred_dir, f[:-4]+'_raw.png')
            print(pred_path)
            gt = load_gt_img(gt_path)
            pred = load_pred_img(pred_path)
            hist += _fast_hist(gt.flatten(), pred.flatten(), n_class)
    return hist


def mat_eval(gt_dir, pred_dir, n_class=2):
    files = os.listdir(pred_dir)
    hist = np.zeros((n_class, n_class))
    for f in files:
        if os.path.splitext(f)[1] == '.mat':
            gt_path = os.path.join(gt_dir, f)
            pred_path = os.path.join(pred_dir, f)
            gt = load_gt_mat(gt_path)
            pred = load_pred_mat(pred_path)
            # from sklearn.metrics import confusion_matrix
            # hist += confusion_matrix(gt.flatten(), pred.flatten(), labels=range(n_class))
            # import matplotlib.pyplot as plt
            # plt.imshow(hist, interpolation='nearest')
            # plt.colorbar()
            # plt.show()
            hist += _fast_hist(gt.flatten(), pred.flatten(), n_class)
            # break
    return hist


def bmp_eval(gt_dir, pred_dir, n_class=2):
    files = os.listdir(pred_dir)
    hist = np.zeros((n_class, n_class))
    for f in files:
        if os.path.splitext(f)[1] == '.bmp':
            gt_path = os.path.join(gt_dir, f[:-4]+'.mat')
            print(gt_path)
            pred_path = os.path.join(pred_dir, f)
            gt = load_gt_mat(gt_path)
            pred = load_bmp_img(pred_path)
            hist += _fast_hist(gt.flatten(), pred.flatten(), n_class)
    return hist


def eval_metrics(hist):
    # seg metrics
    overall_acc = np.diag(hist).sum() / hist.sum()  # overall acc
    acc = np.diag(hist) / hist.sum(axis=1)  # acc for each cls
    mean_acc = np.nanmean(acc)  # mean acc for all cls
    precision = np.diag(hist) / hist.sum(axis=0)  # prec for each cls
    mean_precision = np.nanmean(precision)  # mean prec for all cls
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # IoU for each cls
    mean_iu = np.nanmean(iu)  # mean IoU
    freq = hist.sum(axis=1) / hist.sum()  # frequency
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()  # frequency weighted IoU

    return overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                        filename='traditional_eval.log',
                        format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='w')

    dt_pred_dir = 'D:/documents/data/superparsing/boost_DT/MRF/SemanticLabels'
    nb_pred_dir = 'D:/documents/data/superparsing/naive_bayes/MRF/SemanticLabels'
    hoe_pred_dir = 'D:/documents/data/high_order_energy/mask'
    kittiseg_pred_dir = 'D:/documents/data/kittiseg'
    gt_dir = 'D:/documents/data/superparsing/SemanticLabels'
    hist0 =  mat_eval(gt_dir, nb_pred_dir)
    hist1 = bmp_eval(gt_dir, hoe_pred_dir)
    hist2 = png_eval(kittiseg_pred_dir, kittiseg_pred_dir)

    metrics0 = eval_metrics(hist0)
    metrics1 = eval_metrics(hist1)
    metrics2 = eval_metrics(hist2)

    overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc = metrics0
    logging.info('model_name: {} overall_acc: {:15} acc_0: {:15} acc_1: {:15} mean_acc: {:15} \
        prec_0: {:15} prec_1: {:15} mean_prec: {:15} iu_0: {:15} iu_1: {:15} mean_iu: {:15} fwavacc: {:15}'
        .format('decision tree', overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc))
    
    overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc = metrics1
    logging.info('model_name: {} overall_acc: {:15} acc_0: {:15} acc_1: {:15} mean_acc: {:15} \
        prec_0: {:15} prec_1: {:15} mean_prec: {:15} iu_0: {:15} iu_1: {:15} mean_iu: {:15} fwavacc: {:15}'
        .format('naive bayes', overall_acc, acc[0], acc[1], mean_acc, precision[0], precision[1], mean_precision, iu[0], iu[1], mean_iu, fwavacc))

    overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc = metrics2
    logging.info('model_name: {} overall_acc: {:15} acc: {} mean_acc: {:15} prec: {} mean_prec: {:15} iu: {} mean_iu: {:15} fwavacc: {:15}'
        .format('kittiseg', overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc))