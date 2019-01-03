import numpy as np
from fastai import *
from fastai.vision import *


class ImageItemListCell(ImageItemList):

    def open(self, fn):
        # image = cv2.cvtColor(cv2.imread(str(fn)), cv2.COLOR_BGR2RGB)
        image = np.asarray(PIL.Image.open(fn).convert(self.convert_mode))

        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        sub_image = image[rmin:rmax, cmin:cmax] / 255.

        return Image(pil2tensor(sub_image, np.float32))