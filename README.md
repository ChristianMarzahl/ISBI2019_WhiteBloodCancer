# ISBI2019_WhiteBloodCancer
https://competitionarchitecturerg/competitions/20429

### Results:
#### fived place on the preliminary test set with a score of 0.8764:
Example [Notebook](https://github.com/ChristianMarzahl/ISBI2019_WhiteBloodCancer/blob/master/baseline-resnet18-Patient.ipynb) with a score of 0.8861


# Auto Encoder
- [x] U-Net Architecture
- [x] Pre trained encoder
- [ ] Train as Superpixel and remove skip connections step by step


#  Useful links:

1) Normalisation
https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb

2) Loss
https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

3) Heatmap
https://github.com/swarna04/cs230/blob/master/VisualSemanticModels_Squeezenet1_1.ipynb

4) Pre trained
https://github.com/Cadene/pretrained-models.pytorch

5) Dice

https://nbviewer.jupyter.org/github/daveluo/zanzibar-aerial-mapping/blob/master/znz-segment-buildingfootprint-20181205-comboloss-rn34.ipynb


# Observations:

1) ALL cells are arround 30 percent larger

Id's:
hem = 0
ALL = 1


# Literature
1) Blood Cells - A Practical Guide, 5E (2015)
1.1) acute lymphoblastic leukaemia, 457-60, 458-60, 459


# ToDo:

## Network
- [x] fp_16
- [x] optimal alpha and gamma for focal loss
- [x] U-Net
- [ ] predict rect https://github.com/radekosmulski/whale/blob/master/fluke_detection_redux.ipynb
- [x] Use fpn to classify


## Evaluation
- [x] metric
- [x] submission and ensemble
- [x] Split by Patient
- [x] Split Random
- [x] Split by name like fold via regex
- [ ] copy files to balance hem and all
- [ ] Split by number of patients
- [ ] metrix for each patient
- [ ] evaluation: set all images from one patient to the same label
- [x] Save best model https://docs.fast.ai/callbacks.tracker.html


## Preprocessing
- [x] mean and std fix
- [x] Optimal augmentation search
- [x] use image up scaling [64,128,256,450]
- [x] use mixup
- [x] fit cells to image size
- [ ] cut out and pad smaller to target size
- [x] Cutout https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
- [ ] Auto augment https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
- [x] Stain normalisation and convert all. The images where already stain normalised
- [x] Mixup Parameter


## Statistics
- [x] try to classify cells by size
