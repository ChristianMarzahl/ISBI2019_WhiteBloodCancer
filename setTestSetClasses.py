import os
import csv
from shutil import  copyfile


csv_path = "/server/born_pix/EPA_DATASETS/WhiteBloodCancer/VAL_ISBI_labelfile_Source_reference.csv"
files = "/server/born_pix/EPA_DATASETS/WhiteBloodCancer/test/"
files_target = "/data/Datasets/WhiteBloodCancer/train/fold_3/"

with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(reader):
        if i == 0: #skip header row
            continue
        source_path = os.path.join(files, '{}.bmp'.format(i))
        patient = row[0]
        target_path = os.path.join(files_target, patient)

        copyfile(source_path, target_path)
