from fastai import *
from fastai.vision import *

from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=1.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, targ, reduction='none'):
        t = one_hot_embedding(targ, self.num_classes + 1)
        t = Variable(t[:, :-1].contiguous()).cuda()  # .cpu()
        x = pred[:, :-1]
        w = Variable(self.get_weight(x, t))
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes

    def get_weight(self,x,t):
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)