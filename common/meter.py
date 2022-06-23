import math, numpy as np
import torch

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx  = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions, op_thres=None):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    if op_thres is not None:
        this_class_label[this_class_label>=op_thres] = 1
        this_class_label[this_class_label<op_thres] = 0
    else:
        this_class_label[this_class_label>=threshold_optimal] = 1
        this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore, threshold_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


class Meter:

    def __init__(self):
        self.list   = []
        self.labels = []
        self.preds  = []

    def update(self, item):
        self.list.append(item)
        
    def update_gt(self, label, pred):
        self.labels.append(np.clip(label[0],0,1))
        self.preds.append(pred)
    
    def avg_test(self):
        return torch.tensor(np.array(five_scores(self.labels, self.preds)[0])) * 100.0
    
    def acc_auc(self,thres=None):
        accuracy, auc_value, precision, recall, fscore, thres_op = five_scores(self.labels, self.preds, op_thres=thres)
        acc = torch.tensor(np.array(accuracy)) * 100.0
        auc = torch.tensor(np.array(auc_value)) * 100.0
        return acc, auc, thres_op

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def std(self):
        return torch.tensor(self.list).std() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()