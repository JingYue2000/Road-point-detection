# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:55:24 2021

@author: jingyue zhang
@contact: zhangjingyuezjy@163.com

@description:

@ref: https://blog.csdn.net/weixinhum/article/details/87542937
"""
import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    """
    get confusion matrix
    """
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +label_pred[mask], 
        minlength=n_class ** 2).reshape(n_class, n_class)
#    from sklearn.metrics import confusion_matrix
#    hist = confusion_matrix(label_true, label_pred, labels=range(n_class))
#    import matplotlib.pyplot as plt
#    plt.imshow(hist, interpolation='nearest')
#    plt.colorbar()
#    plt.show()
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
#    hist = _fast_hist(label_trues.flatten(), label_preds.flatten(), n_class)
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    overall_acc = np.diag(hist).sum() / hist.sum() # overall acc
    acc = np.diag(hist) / hist.sum(axis=1) # acc for each cls
    mean_acc = np.nanmean(acc) # mean acc for all cls
    precision = np.diag(hist) / hist.sum(axis=0) # prec for each cls
    mean_precision = np.nanmean(precision) # mean prec for all cls
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) # IoU for each cls
    mean_iu = np.nanmean(iu) # mean IoU
    freq = hist.sum(axis=1) / hist.sum() # frequency
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()#frequency weighted IoU
    return overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc

if __name__ == '__main__':
    label_true=np.array([[[0,0,0,0,0,4], [0,0,0,2,0,0]],[[0,0,0,0,0,4], [0,0,0,2,0,0]]])
    label_pred=np.array([[[0,0,0,0,0,4], [0,0,0,2,0,0]],[[0,0,0,0,0,1], [0,0,0,2,0,0]]])
    
    overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc = label_accuracy_score(label_true, label_pred, 5)
    print(overall_acc, acc, mean_acc, precision, mean_precision, iu, mean_iu, fwavacc)

