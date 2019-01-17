import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


from fastai import *
from fastai.vision import *
from fastai.callbacks import *


class F1Weighted(Callback):

    def on_epoch_begin(self, **kwargs):
        self.y_true, self.y_pred = [], []

    def on_batch_end(self, last_output, last_target, **kwargs):
        if isinstance(last_output, list):
            preds = last_output[1].argmax(1)
            if last_target[1].shape[1] == 2:
                target = last_target[1].squeeze()
            else:
                target = last_target[1][:, :2].argmax(1)
            self.y_pred.extend(preds.data.cpu().numpy())
            self.y_true.extend(target.data.cpu().numpy())
        else:
            preds = last_output.argmax(1)
            self.y_pred.extend(preds.data.cpu().numpy())
            self.y_true.extend(last_target.data.cpu().numpy())

    def on_epoch_end(self, **kwargs):
        self.metric = f1_score(self.y_true, self.y_pred, average='weighted')


class MCC(Callback):
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    '''

    def on_epoch_begin(self, **kwargs):
        self.y_true, self.y_pred = [], []

    def on_batch_end(self, last_output, last_target, **kwargs):
        if isinstance(last_output, list):
            preds = last_output[1].argmax(1)
            if last_target[1].shape[1] == 2:
                target = last_target[1].squeeze()
            else:
                target = last_target[1][:, :2].argmax(1)
            self.y_pred.extend(preds.data.cpu().numpy())
            self.y_true.extend(target.data.cpu().numpy())
        else:
            preds = last_output.argmax(1)
            self.y_pred.extend(preds.data.cpu().numpy())
            self.y_true.extend(last_target.data.cpu().numpy())

    def on_epoch_end(self, **kwargs):
        self.metric = matthews_corrcoef(self.y_true, self.y_pred)


def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

def area(boxes):
    return ((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))

def union(preds, targs):
    return area(preds) + area(targs) - intersection(preds, targs)

def IoU(preds, targs):
    return intersection(preds, targs) / union(preds, targs)