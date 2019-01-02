# ISBI2019_WhiteBloodCancer
https://competitions.codalab.org/competitions/20429


#  Useful links:

1) Normalisation
https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

2) Loss
https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

# Observations:

1) All cells are arround 30 percent larger

# Literature
1) Blood Cells - A Practical Guide, 5E (2015)
1.1) acute lymphoblastic leukaemia, 457-60, 458-60, 459


# ToDo:

## Network
- [x] fp_16
- [ ] optimal alpha and gamma for focal loss
- [ ] U-Net

## Evaluation
- [x] metric
- [ ] submission
- [x] Split by Patient
- [x] Split Random
- [x] Split by name like fold via regex
- [ ] copy files to balance hem and all
- [ ] Split by number of patients
- [ ] metrix for each patient
- [ ] evaluation: set all images from one patient to the same label
- [ ] Save best model https://docs.fast.ai/callbacks.tracker.html

## Preprocessing
- [x] mean and std fix
- [ ] Optimal augmentation
- [ ] use image up scaling [64,128,256,450]
- [x] use mixup
- [x] fit cells to image size
- [ ] Cutout https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
- [ ] Auto augment https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py

## Statistics
- [x] try to classify cells by size
