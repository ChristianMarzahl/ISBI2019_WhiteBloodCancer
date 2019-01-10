import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm


import staintools

def read_image(fn):

    im = cv2.imread(str(fn))
    # Convert from cv2 standard of BGR to our convention of RGB.
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



path = '/data/Datasets/WhiteBloodCancer/norm/'
fnames = list(Path(path).glob('**/*.bmp'))

# Use UID_1_1_1_all.bmp als ref
target_name = [fn for fn in fnames if "UID_1_1_1_all.bmp" in str(fn)][0]
target = read_image(target_name)

normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)

for fn in tqdm(fnames):

    to_transform = read_image(fn)
    mask = to_transform > 0

    to_transform = normalizer.transform(to_transform)
    to_transform = to_transform * mask

    cv2.imwrite(str(fn), to_transform)



