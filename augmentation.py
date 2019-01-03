import numpy as np
from fastai import *
from fastai.vision import *

def cutout(img, n_holes:int = 5, length:float = 0.05)->Tensor:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (float): The length (in percent of the image size) of each square patch.
    """
    h = img.size(1)
    w = img.size(2)
    patch_width = int(w * length)
    path_height = int(h * length)

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - path_height // 2, 0, h)
        y2 = np.clip(y + path_height // 2, 0, h)
        x1 = np.clip(x - patch_width // 2, 0, w)
        x2 = np.clip(x + patch_width // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    return img * mask