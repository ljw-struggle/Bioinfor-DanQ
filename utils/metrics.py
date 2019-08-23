# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics

def calculate_auroc(predictions, labels):
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan

    return fpr_list, tpr_list, auroc

def calculate_aupr(predictions, labels):
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.auc(recall_list, precision_list)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr