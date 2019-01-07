from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


from fastai import *
from fastai.vision import *
from fastai.callbacks import *


class F1Weighted(Callback):

    def on_epoch_begin(self, **kwargs):
        self.y_true, self.y_pred = [], []

    def on_batch_end(self, last_output, last_target, **kwargs):
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
        preds = last_output.argmax(1)
        self.y_pred.extend(preds.data.cpu().numpy())
        self.y_true.extend(last_target.data.cpu().numpy())

    def on_epoch_end(self, **kwargs):
        self.metric = matthews_corrcoef(self.y_true, self.y_pred)