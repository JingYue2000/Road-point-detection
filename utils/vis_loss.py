#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-7-13 19:32:16
# @Author  : jingyue zhang
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from util import config


def max_min_norm(x):
    mx = np.max(x)
    mn = np.min(x)
    return (x - mn) / (mx - mn)
    
def z_score_norm(x):
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / sigma
    
def sigmoid(x):
    return 1.0 / (1 + np.exp(-np.array(x)))

def parse_train_log(log_file):
    model_names = []
    loss, loss_coor = [], []
    overall_acc = []
    acc_0, acc_1, mean_acc = [], [], []
    iu_0, iu_1, miou = [], [], []
    prec_0, prec_1, mean_prec = [], [], []
    fwvacc = []
    dist1 , dist2 = [], []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            logs = line.split()
            # print(logs[9])
            # print(logs)
            # print(len(logs))
            if len(logs) > 15 and logs[9] == 'lr:':
                model_names.append(logs[10])
                loss.append(float(logs[12]))
                loss_coor.append(float(logs[14]))
                overall_acc.append(float(logs[16]))
                acc_0.append(float(logs[18]))
                acc_1.append(float(logs[20]))
                mean_acc.append(float(logs[22]))
                prec_0.append(float(logs[24]))
                prec_1.append(float(logs[26]))
                mean_prec.append(float(logs[28]))
                iu_0.append(float(logs[30]))
                iu_1.append(float(logs[32]))
                miou.append(float(logs[34]))
                fwvacc.append(float(logs[36]))
                dist1.append(float(logs[38]))
                dist2.append(float(logs[40]))
    print(len(loss))
    k = len(loss)
    t = np.arange(0,k)
    s_var = dist1
    print('avg={:.3f} max={:.3f} min={:.3f} std={:.3f}'.format(np.mean(s_var), np.max(s_var), np.min(s_var), np.std(s_var)))
    metrics = ['road IoU', 'road prec', 'road acc']
    
    plt.figure(0)
    plt.plot(miou)
    # plt.savefig('loss.pdf', format='pdf', transparent=False, dpi=300)
    plt.show()

def parse_log(log_file):
    model_names = []
    loss, loss_coor = [], []
    overall_acc = []
    acc_0, acc_1, mean_acc = [], [], []
    iu_0, iu_1, miou = [], [], []
    prec_0, prec_1, mean_prec = [], [], []
    fwvacc = []
    dist1 , dist2 = [], []
    with open(log_file, 'r') as f:
        for line in f.readlines():
            logs = line.split()
            print(logs[7])
            print(len(logs))
            if len(logs) > 15 and logs[6] == 'model_name:':
                model_names.append(logs[7])
                loss.append(float(logs[9]))
                loss_coor.append(float(logs[11]))
                overall_acc.append(float(logs[13]))
                acc_0.append(float(logs[15]))
                acc_1.append(float(logs[17]))
                mean_acc.append(float(logs[19]))
                prec_0.append(float(logs[21]))
                prec_1.append(float(logs[23]))
                mean_prec.append(float(logs[25]))
                iu_0.append(float(logs[27]))
                iu_1.append(float(logs[29]))
                miou.append(float(logs[31]))
                fwvacc.append(float(logs[33]))
                dist1.append(float(logs[35]))
                dist2.append(float(logs[37]))
    print(len(loss))
    # t = np.array(list(range(2,13,1)))
    # t = np.arange(2,13,1)
    k = len(loss)//2
    t = np.arange(0,k,1)
    print(t)
    s_var = dist1
    print('avg={:.3f} max={:.3f} min={:.3f} std={:.3f}'.format(np.mean(s_var), np.max(s_var), np.min(s_var), np.std(s_var)))
    metrics = ['road IoU', 'road prec', 'road acc']
    
    plt.figure(0)
    plt.plot(t, iu_1[:k], 'r-*')
    plt.plot(t, overall_acc[:k], 'r-d')
    plt.plot(t, prec_1[:k], 'r-s')
    plt.plot(t, acc_1[:k], 'r-^')
    plt.plot(t, iu_1[k:], 'g-*')
    plt.plot(t, overall_acc[k:], 'g-d')
    plt.plot(t, prec_1[k:], 'g-s')
    plt.plot(t, acc_1[k:], 'g-^')
    plt.plot(t, prec_1[:k], 'r->')
    plt.plot(t, prec_1[k:2*k], 'g->')
    # plt.plot(t, prec_1[2*k:], 'b-s')
    plt.plot(t, miou[:k], 'r-<')
    plt.plot(t, miou[k:2*k], 'g-<')    
    # plt.plot(t, miou[2*k:], 'b-s')


    plt.xlabel('# points')
    plt.ylabel('metrics(%)')
    # plt.legend(['0iter+w/o part', '0iter+w part', '3iter+w part'])
    # plt.savefig('iter_part.pdf', format='pdf', transparent=False, dpi=300)
    plt.show()
    
    # plt.figure(1)
    # bar_width = 0.2
    # labels = ['mIoU', 'acc', 'prec', 'fwvacc']
    # fig, ax = plt.subplots()
    # plt.bar(t, miou[k:], bar_width, label=labels[0])
    # plt.bar(t+2*bar_width, mean_acc[k:], bar_width, label=labels[1])
    # plt.bar(t+3*bar_width, mean_prec[k:], bar_width, label=labels[2])
    # plt.bar(t+bar_width, fwvacc[k:], bar_width, label=labels[3])
    # # plt.bar(t, miou[:k], bar_width, label=labels[0])
    # # plt.bar(t+2*bar_width, mean_acc[:k], bar_width, label=labels[1])
    # # plt.bar(t+3*bar_width, mean_prec[:k], bar_width, label=labels[2])
    # # plt.bar(t+bar_width, fwvacc[:k], bar_width, label=labels[3])
    # plt.legend(labels)
    # plt.xticks(t+1.5*bar_width,t)
    # ax.set_ylim(0.94, 1.)
    # plt.xlabel('# iters')
    # plt.ylabel('metrics(%)')
    # #plt.savefig('iter-effect-bar.pdf', format='pdf', transparent=False, dpi=300)
    # plt.show()


def parse_act_fun_log(log_file):
    loss = []
    loss_coor = []
    lr = []
    index = 0
    with open(log_file, 'r') as f:
        for line in f.readlines():
            logs = line.split()
            if len(logs) > 15 and logs[10] == 'lr:' and logs[12] == 'loss:' and logs[14] == 'loss_coor:':
                lr.append(float(logs[11]))
                loss.append(float(logs[13]))
                loss_coor.append(float(logs[15]))
            index += 1
    print(len(loss))
    var = loss_coor
    var = signal.medfilt(volume=var, kernel_size=51)
    # s_var = var[len(var)*2//3:]
    # print('avg={:.3f} max={:.3f} min={:.3f} std={:.3f}'.format(np.mean(s_var), np.max(s_var), np.min(s_var), np.std(s_var)))
    #plt.figure(0)
    # plt.plot(var[:100], '-*')
    # plt.plot(signal.medfilt(volume=var, kernel_size=51)) # smooth line
    return var
    #plt.close(0)
    #plt.show()


# if __name__ == '__main__':
#     funcs = ['softmax', 'leakyReLU', 'tanh', 'SELU', 'CELU', 'swish', 'linear']
#     style = ['-', '-.', '-s', '-d', '--', '-*', '->']
#     step, end = 123, 12300
#     for i, func in enumerate(funcs):
#         print(func)
#         log_file = 'models/8_points-{}-with_part-fix_heatmap-models/train.log'.format(func)
#         if func == 'linear':
#             log_file = 'models/8_points-with_part-fix_heatmap-models/train.log'
#         if os.path.exists(log_file):
#             var = parse_act_fun_log(log_file)
#             if i == 6:
#                 step = 62
#                 end = 6200
#             plt.plot(range(100), var[:end:step], style[i])
#     plt.legend(funcs)
#     plt.xlabel('# epoch')
#     plt.ylabel('coordinate error')
#     plt.savefig('loss_func.pdf', format='pdf', transparent=False)
#     plt.show()


if __name__ == '__main__':
    log_file = '../iters_summary_eval.log'
    #log_file = '../models/8-points_3-iters_True-part_cat-fusion_models/train.log'
    parse_log(log_file)
