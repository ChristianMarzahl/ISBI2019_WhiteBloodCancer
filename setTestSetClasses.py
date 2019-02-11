import os
import csv
from shutil import  copyfile


csv_path = "/server/born_pix/EPA_DATASETS/WhiteBloodCancer/isbi_valid_GT.csv"
files = "/server/born_pix/EPA_DATASETS/WhiteBloodCancer/test/"
files_target = "/data/Datasets/WhiteBloodCancer/test_gt/"

with open(csv_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='"')
    for i, row in enumerate(reader):
        i += 1
        source_path = os.path.join(files, '{}.bmp'.format(i))
        patient = 'UID_H99_1_{}_hem.bmp'.format(i) if int(row[0]) == 0 else 'UID_99_1_{}_all.bmp'.format(i)
        target_path = os.path.join(files_target, patient)

        copyfile(source_path, target_path)
