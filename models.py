import numpy as np

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *


class WhiteBlodNet(nn.Module):

    def __init__(self, encoder: nn.Module, **kwargs):
        super().__init__()

        imsize = (256,256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(self._get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs])
        x = dummy_eval(encoder, imsize).detach()

        self.encoder = encoder
        self.aap_flatten = nn.Sequential(*[AdaptiveConcatPool2d(), Flatten()])

        self.nf = (x.shape[1] + sum([hook.stored.shape[1] for hook in self.sfs])) * 2

    def forward(self, x):

        x = [self.aap_flatten(self.encoder(x))] + [self.aap_flatten(flat.stored) for flat in self.sfs]
        x = torch.cat(x, dim=1)

        return x


    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

    def _get_sfs_idxs(self, sizes: Sizes) -> List[int]:
        "Get the indexes of the layers where the size of the activation changes."
        feature_szs = [size[-1] for size in sizes]
        sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
        if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
        return sfs_idxs



class WhiteBlodHead(nn.Module):
    def __init__(self, nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False):
        "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
        super().__init__()
        self.bn_final = bn_final
        self.nc = nc

        self.lin_ftrs = [nf, 512, self.nc] if lin_ftrs is None else [nf] + lin_ftrs + [self.nc]

        self.ps = listify(ps)
        if len(self.ps) == 1: self.ps = [self.ps[0] / 2] * (len(self.lin_ftrs) - 2) + self.ps
        self.actns = [nn.ReLU(inplace=True)] * (len(self.lin_ftrs) - 2) + [None]

        layers = []
        for ni, no, p, actn in zip(self.lin_ftrs[:-1], self.lin_ftrs[1:], self.ps, self.actns):
            layers += bn_drop_lin(ni, no, True, p, actn)

        self.final = nn.Sequential(*layers)

    def forward(self, x):

        return self.final(x)